// visualization.cpp

// external
#define GLFW_INCLUDE_NONE
#include <algorithm>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <spdlog/spdlog.h>
#include <stdexcept>

// internal
#include "source/base/helpers.h"
#include "visualization.h"



using namespace Vis;

VisualizationMgr::VisualizationMgr(Sim::SimulationMgr& simulationMgr,
								   Bench::BenchmarksTracker& benchmarksTracker,
								   Sim::SimParamsMgr& simParamsMgr,
								   Vis::VisParamsMgr& visParamsMgr)
	: simulationMgr_{ simulationMgr }
	, benchmarksTracker_{ benchmarksTracker }
	, simParamsMgr_{ simParamsMgr }
	, visParamsMgr_{ visParamsMgr }
	, userInputMgr_{ simParamsMgr, visParamsMgr }
	, userInterfaceMgr_{ simParamsMgr, visParamsMgr }
{
}

void VisualizationMgr::Init(const WindowConfig& windowConfig)
{
	if (!glfwInit())
	{
		throw std::runtime_error("Failed to initialize GLFW.");
	}

	// initial snapshot
	SyncParameters(true);
	stopRequested_.store(false, std::memory_order_release);

	// window and GL context creation
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, VisConstants::glVersion.major);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, VisConstants::glVersion.minor);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	// glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GLFW_TRUE); // TODO: disable for release?
	glfwWindowHint(GLFW_RESIZABLE, visParams_.window.resizable ? GLFW_TRUE : GLFW_FALSE);
	glfwWindowHint(GLFW_SAMPLES, 4);

	window_ = glfwCreateWindow(visParams_.window.size.x,
							   visParams_.window.size.y,
							   visParams_.window.title,
							   nullptr, nullptr);

	if (!window_)
	{
		glfwTerminate();
		throw std::runtime_error("Failed to create GLFW window.");
	}

	// TODO: choose window position properly
	GLFWmonitor* monitor = glfwGetPrimaryMonitor();
	const GLFWvidmode* mode = monitor ? glfwGetVideoMode(monitor) : nullptr;
	if (mode)
	{
		const glm::ivec2 monitorSize{ mode->width, mode->height };
		glfwSetWindowPos(window_,
			monitorSize.x / 2 - visParams_.window.size.x / 2 + std::max(0, monitorSize.x - 2800),
			monitorSize.y / 2 - visParams_.window.size.y / 2);
	}

	glfwMakeContextCurrent(window_);
	glfwSwapInterval(windowConfig.vsync ? 1 : 0);

	// TODO: use gladLoadGL() instead ?
	if (!gladLoadGLLoader(GLADloadproc(glfwGetProcAddress)))
	{
		glfwDestroyWindow(window_);
		window_ = nullptr;
		glfwTerminate();
		throw std::runtime_error("Failed to initialize GLAD.");
	}

	// set GL functions and states
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
	glEnable(GL_CULL_FACE);
	glCullFace(GL_BACK);
	glEnable(GL_MULTISAMPLE);

	// sizes and dpi
	int fbWidth, fbHeight, winWidth, winHeight;
	float xScale = 1.0f, yScale = 1.0f;
	glfwGetFramebufferSize(window_, &fbWidth, &fbHeight);
	glfwGetWindowSize(window_, &winWidth, &winHeight);
	glfwGetWindowContentScale(window_, &xScale, &yScale);

	visParamsMgr_.SetParam(FrameBufferPatch
	{
		.size = glm::ivec2{ fbWidth, fbHeight },
		.dpiScale = xScale >= yScale ? xScale : yScale
	});
	visParamsMgr_.SetParam(WindowPatch
	{
		.size = glm::ivec2{ winWidth, winHeight },
	});

	// update snapshot
	SyncParameters(true);

	// overwrite callbacks with own ones
	glfwSetWindowUserPointer(window_, this);
	glfwSetFramebufferSizeCallback(window_, &VisualizationMgr::GLFW_FramebufferSizeCB);
	glfwSetWindowSizeCallback(window_, &VisualizationMgr::GLFW_WindowSizeCB);
	glfwSetWindowContentScaleCallback(window_, &VisualizationMgr::GLFW_WindowContentScaleCB);
	glfwSetKeyCallback(window_, &VisualizationMgr::GLFW_KeyCB);
	glfwSetMouseButtonCallback(window_, &VisualizationMgr::GLFW_MouseButtonCB);
	glfwSetCursorPosCallback(window_, &VisualizationMgr::GLFW_CursorPosCB);
	glfwSetScrollCallback(window_, &VisualizationMgr::GLFW_ScrollCB);

	// initialize sub-system managers
	userInputMgr_.Init(visParams_);
	userInterfaceMgr_.Init(window_, simParams_, VisConstants::glVersion.name);
	renderingMgr_.Init();

	// initial GL viewport
	// TODO: outsource to renderingMgr
	glViewport(0, 0, fbWidth, fbHeight);

}

void VisualizationMgr::Shutdown()
{
	if (!window_) { return; }

	// first shut down sub-systems
	userInputMgr_.Shutdown();
	userInterfaceMgr_.Shutdown();
	renderingMgr_.Shutdown();

	glfwDestroyWindow(window_);
	window_ = nullptr;
	glfwTerminate();
}

void VisualizationMgr::RunVis()
{
	if (!window_) { return; }

	double lastTime = glfwGetTime();

	while (!glfwWindowShouldClose(window_) &&
		   !stopRequested_.load(std::memory_order_acquire))
	{
		// process input events and parameter changes
		glfwPollEvents();
		SyncParameters();

		// time step for handling continuous input
		double now = glfwGetTime();
		float dt = static_cast<float>(now - lastTime);
		lastTime = now;
		userInputMgr_.UpdateCamera(visParams_.camera, visParams_.frameBuffer, dt);

		// fetch latest visual snap and replace vis-specific simulation state
		SyncVisSimState();

		// fetch latest bench and stat snaps to make local copies
		SyncBenchSnap();
		SyncStatSnap();

		// 1. begin new ImGui frame
		userInterfaceMgr_.BeginFrame();

		// 2. draw 3D scene given the current state and camera
		// TODO: extend to also take the view?
		glClearColor(visParams_.render.clearColor.x,
					 visParams_.render.clearColor.y,
					 visParams_.render.clearColor.z,
					 visParams_.render.clearColor.w);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		renderingMgr_.Render(visParams_.render, visParams_.camera);

		// 3. build ImGui panels
		userInterfaceMgr_.BuildUserInterfaceSections(simulationMgr_,
			localBenchSnap_.get(), localStatSnap_.get(), simParams_, visParams_);
		userInterfaceMgr_.EndFrame();

		// 4. render ImGui on top of the 3D scene
		userInterfaceMgr_.RenderDrawData();

		glfwSwapBuffers(window_);
	}
}

void VisualizationMgr::RequestStop()
{
	stopRequested_.store(true, std::memory_order_release);
}

// ===== callback trampolines =====

void VisualizationMgr::GLFW_FramebufferSizeCB(GLFWwindow* window, int width, int height)
{
	if (auto* self = static_cast<VisualizationMgr*>(glfwGetWindowUserPointer(window)))
	{
		self->OnFramebufferSize(width, height);
	}
}

void VisualizationMgr::GLFW_WindowSizeCB(GLFWwindow* window, int width, int height)
{
	if (auto* self = static_cast<VisualizationMgr*>(glfwGetWindowUserPointer(window)))
	{
		self->OnWindowSize(width, height);
	}
}

void VisualizationMgr::GLFW_WindowContentScaleCB(GLFWwindow* window, float xScale, float yScale)
{
	if (auto* self = static_cast<VisualizationMgr*>(glfwGetWindowUserPointer(window)))
	{
		self->OnWindowContentScale(xScale, yScale);
	}
}

void VisualizationMgr::GLFW_KeyCB(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	if (auto* self = static_cast<VisualizationMgr*>(glfwGetWindowUserPointer(window)))
	{
		self->OnKey(key, scancode, action, mods);
	}
}

void VisualizationMgr::GLFW_MouseButtonCB(GLFWwindow* window, int button, int action, int mods)
{
	if (auto* self = static_cast<VisualizationMgr*>(glfwGetWindowUserPointer(window)))
	{
		self->OnMouseButton(button, action, mods);
	}
}

void VisualizationMgr::GLFW_CursorPosCB(GLFWwindow* window, double xPos, double yPos)
{
	if (auto* self = static_cast<VisualizationMgr*>(glfwGetWindowUserPointer(window)))
	{
		self->OnCursorPos(xPos, yPos);
	}
}

void VisualizationMgr::GLFW_ScrollCB(GLFWwindow* window, double xOff, double yOff)
{
	if (auto* self = static_cast<VisualizationMgr*>(glfwGetWindowUserPointer(window)))
	{
		self->OnScroll(xOff, yOff);
	}
}

// ===== callback instance handlers =====

void VisualizationMgr::OnFramebufferSize(int width, int height)
{
	visParamsMgr_.SetParam(FrameBufferPatch
	{ 
		.size = glm::ivec2{ width, height }
	});

	glViewport(0, 0, width, height);
}

void VisualizationMgr::OnWindowSize(int width, int height)
{
	visParamsMgr_.SetParam(WindowPatch
	{
		.size = glm::ivec2{ width, height }
	});
}

void VisualizationMgr::OnWindowContentScale(float xScale, float yScale)
{
	if (!Base::Float_AlmostEqual(xScale, yScale))
	{
		SPDLOG_WARN("X and Y scales differ while determining DPI.");
	}

	float dpiScale = xScale >= yScale ? xScale : yScale;

	visParamsMgr_.SetParam(FrameBufferPatch
	{
		.dpiScale = dpiScale
	});

	// TODO: make other managers react to dpi change
	userInterfaceMgr_.OnDPIScaleChanged(dpiScale);
}

// pressed keys processed by user interface first
// if not handled, then given by user input manager second
void VisualizationMgr::OnKey(int key, int scancode, int action, int mods)
{
	// TODO: UI gets first shot; if not handled, forward to input mgr
	if (!userInterfaceMgr_.OnKey(key, scancode, action, mods))
	{
		userInputMgr_.OnKey(simParams_, visParams_.camera, visParams_.render,
			key, scancode, action, mods);
	}
}

// pressed mouse buttons processed by user interface first
// if not handled, then given by user input manager second
void VisualizationMgr::OnMouseButton(int button, int action, int mods)
{
	// TODO: UI gets first shot; if not handled, forward to input mgr
	if (!userInterfaceMgr_.OnMouseButton(button, action, mods))
	{
		userInputMgr_.OnMouseButton(button, action, mods);
	}
}

// mouse movement processed by user interface first
// if not handled, then given by user input manager second
void VisualizationMgr::OnCursorPos(double xPos, double yPos)
{
	// TODO: UI gets first shot; if not handled, forward to input mgr
	if (!userInterfaceMgr_.OnCursorPos(xPos, yPos))
	{
		userInputMgr_.OnCursorPos(xPos, yPos);
	}
}

// mouse scrolling buttons processed by user interface first
// if not handled, then given by user input manager second
void VisualizationMgr::OnScroll(double xOff, double yOff)
{
	// TODO: UI gets first shot; if not handled, forward to input mgr
	if (!userInterfaceMgr_.OnScroll(xOff, yOff))
	{
		userInputMgr_.OnScroll(xOff, yOff);
	}
}

void VisualizationMgr::SyncParameters(bool force)
{
	uint64_t latestSimParamsVersion = simParamsMgr_.StableVersion();
	if (latestSimParamsVersion != simParamsVersion_ ||
		force)
	{
		auto snapshot = simParamsMgr_.SnapshotParams();

		// TODO: remove debug log
		// SPDLOG_INFO("Synced 'simParams' snapshot from version {} to {}",
		// simParamsVersion_, std::get<1>(snapshot));

		simParams_ = std::get<0>(snapshot);
		simParamsVersion_ = std::get<1>(snapshot);
	}

	uint64_t latestVisParamsVersion = visParamsMgr_.StableVersion();
	if (latestVisParamsVersion != visParamsVersion_ ||
		force)
	{
		auto snapshot = visParamsMgr_.SnapshotParams();

		// TODO: remove debug log
		// SPDLOG_INFO("Synced 'visParams' snapshot from version {} to {}",
		// visParamsVersion_, std::get<1>(snapshot));

		visParams_ = std::get<0>(snapshot);
		visParamsVersion_ = std::get<1>(snapshot);
		simulationMgr_.SetVisualScalarMode(visParams_.render.scalarModeF);
	}
}

void VisualizationMgr::SyncVisSimState()
{
	if (!visualSnapMailbox_)
	{
		SPDLOG_ERROR("Failed to synchronize visual simulation state (no mailbox set).");
		return;
	}

	bool hasFrameCUDA = false;
	#ifdef TURBOSPHARTICLES_WITH_CUDA
		Sync::VisualFrameReadCUDA frameCUDA{};
		if (visualSnapMailbox_->GetFrameMailboxCUDA().TryFetchLatest(frameCUDA))
		{
			hasFrameCUDA = true;
		}
	#endif

	// try the host mailbox when no CUDA frame is available
	std::shared_ptr<Sync::VisualSnapshot> snap;
	#ifdef TURBOSPHARTICLES_WITH_CUDA
		if (hasFrameCUDA) { snap = frameCUDA.auxiliarySnap; }
	#endif
	if (!hasFrameCUDA && !visualSnapMailbox_->TryFetchLatestSnapshot(snap))
	{
		// no new snapshot available
		return;
	}
	if (!snap)
	{
		SPDLOG_ERROR("Failed to synchronize visual simulation state (empty snapshot).");
		#ifdef TURBOSPHARTICLES_WITH_CUDA
			if (hasFrameCUDA)
			{
				renderingMgr_.UpdateParticlesCUDA(frameCUDA, simParams_, visParams_);
			}
		#endif
		return;
	}

	// move snapshot data into back buffers
	view_.simStep = snap->simStep;
	posF_[back_].swap(snap->posF);
	scalarDataF_[back_].swap(snap->scalarDataF);
	posB_[back_].swap(snap->posB);
	volB_[back_].swap(snap->volB);
	boxesB_[back_].swap(snap->boxesB);
	posR_[back_].swap(snap->posR);
	volR_[back_].swap(snap->volR);
	centerOfMassRB_[back_].swap(snap->centerOfMassRB);
	orientationRB_[back_].swap(snap->orientationRB);
	scalarMode_ = snap->scalarModeF;

	// ensure scalar data size matches position size
	if (scalarMode_ != Base::ColorCodingMode::None)
	{
		if (scalarDataF_[back_].size() != posF_[back_].size())
		{
			SPDLOG_WARN("Scalar count {} does not match position count {} for color coding.",
				scalarDataF_[back_].size(), posF_[back_].size());

			// fix by resizing and filling with zeros
			scalarDataF_[back_].resize(posF_[back_].size(), 0.0f);
		}
	}
	else // no color coding -> keep scalar buffer empty
	{
		scalarDataF_[back_].clear();
	}

	SwapVisBuffers();

	// pack the read-only view for the rendering manager to use it
	view_.N_F = static_cast<uint32_t>(posF_[front_].size());
	view_.posF = std::span<const glm::vec3>(posF_[front_].data(), posF_[front_].size());
	view_.scalarDataF = (scalarMode_ == Base::ColorCodingMode::None)
		? std::span<const float>()
		: std::span<const float>(scalarDataF_[front_].data(), scalarDataF_[front_].size());
	view_.scalarModeF = scalarMode_;
	view_.N_B = static_cast<uint32_t>(posB_[front_].size());
	view_.posB = std::span<const glm::vec3>(posB_[front_].data(), posB_[front_].size());
	view_.volB = std::span<const float>(volB_[front_].data(), volB_[front_].size());
	view_.boxesB = std::span<const Scene::OBB>(boxesB_[front_].data(), boxesB_[front_].size());
	view_.N_R = static_cast<uint32_t>(posR_[front_].size());
	view_.posR = std::span<const glm::vec3>(posR_[front_].data(), posR_[front_].size());
	view_.volR = std::span<const float>(volR_[front_].data(), volR_[front_].size());
	view_.N_RB = static_cast<uint32_t>(centerOfMassRB_[front_].size());
	view_.centerOfMassRB = std::span<const glm::vec3>(
		centerOfMassRB_[front_].data(), centerOfMassRB_[front_].size());
	view_.orientationRB = std::span<const glm::mat3>(
		orientationRB_[front_].data(), orientationRB_[front_].size());

	// local copy for thread safety
	const auto simParams = simParams_;
	const auto visParams = visParams_;

	// push to GPU to let renderer use the new data
	#ifdef TURBOSPHARTICLES_WITH_CUDA
		if (hasFrameCUDA)
		{
			renderingMgr_.UpdateParticlesCUDA(frameCUDA, simParams, visParams);
		}
		else
	#endif
	{
		renderingMgr_.UpdateParticles(view_, simParams, visParams);
	}
	renderingMgr_.UpdateCentersOfMass(view_, simParams);
	renderingMgr_.UpdateWireframes(view_);
}

void VisualizationMgr::SyncBenchSnap()
{
	if (!benchSnapMailbox_)
	{
		SPDLOG_ERROR("Failed to synchronize benchmark snapshot (no mailbox set).");
		return;
	}

	// try to fetch latest snapshot for a local copy
	std::shared_ptr<Sync::BenchSnapshot> snap;
	if (benchSnapMailbox_->TryFetchLatestSnapshot(snap))
	{
		localBenchSnap_ = std::move(snap);
	}
}

void VisualizationMgr::SyncStatSnap()
{
	if (!statSnapMailbox_)
	{
		SPDLOG_ERROR("Failed to synchronize statistics snapshot (no mailbox set).");
		return;
	}

	// try to fetch latest snapshot for a local copy
	std::shared_ptr<Sync::StatSnapshot> snap;
	if (statSnapMailbox_->TryFetchLatestSnapshot(snap))
	{
		localStatSnap_ = std::move(snap);
	}
}

void VisualizationMgr::SwapVisBuffers() { std::swap(front_, back_); }
