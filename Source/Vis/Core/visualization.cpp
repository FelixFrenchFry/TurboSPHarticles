// visualization.cpp

// external
#define GLFW_INCLUDE_NONE
#include <algorithm>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <random>
#include <spdlog/spdlog.h>

// internal
#include "../../Base/helpers.h"
#include "visualization.h"



using namespace Vis;

VisualizationMgr::VisualizationMgr(Sim::SimParamsMgr& simParamsMgr,
								   Vis::VisParamsMgr& visParamsMgr)
	: simParamsMgr_{ simParamsMgr }
	, visParamsMgr_{ visParamsMgr }
	, simParams_{}
	, visParams_{}
	, userInputMgr_{ simParamsMgr, visParamsMgr }
	, userInterfaceMgr_{ simParamsMgr, visParamsMgr }
	, renderingMgr_{ simParamsMgr, visParamsMgr }
{
	// handled by Init()
}

bool VisualizationMgr::Init(const WindowConfig& windowConfig)
{
	if (!glfwInit())
	{
		SPDLOG_ERROR("Failed to initialize GLFW");
		return false;
	}

	// initial snapshot
	SyncParameters(true);

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
		SPDLOG_ERROR("Failed to create GLFW window");
		glfwTerminate();
		return false;
	}

	glfwMakeContextCurrent(window_);
	glfwSwapInterval(windowConfig.vsync ? 1 : 0);

	// TODO: use gladLoadGL() instead ?
	if (!gladLoadGLLoader(GLADloadproc(glfwGetProcAddress)))
	{
		SPDLOG_ERROR("Failed to initialize GLAD");
		glfwDestroyWindow(window_);
		glfwTerminate();
		return false;
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
	userInterfaceMgr_.Init(window_, VisConstants::glVersion.name);
	renderingMgr_.Init(visParams_.frameBuffer.size, visParams_.frameBuffer.dpiScale);

	// pre-reserve double-buffers
	// TODO: use better estimate of max sizes
	const std::int64_t N_reserve = 100'000;
	position_[0].reserve(N_reserve);
	position_[1].reserve(N_reserve);
	scalarData_[0].reserve(N_reserve);
	scalarData_[1].reserve(N_reserve);

	// initial GL viewport
	// TODO: outsource to renderingMgr
	glViewport(0, 0, fbWidth, fbHeight);

	return true;
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

	while (!glfwWindowShouldClose(window_))
	{
		// process input events and parameter changes
		glfwPollEvents();
		SyncParameters();

		// time step for handling continuous input
		double now = glfwGetTime();
		float dt = static_cast<float>(now - lastTime);
		lastTime = now;
		userInputMgr_.UpdateCamera(visParams_.camera, visParams_.frameBuffer, dt);

		// get latest vis-specific simulation state
		SyncVisSimState();

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
		userInterfaceMgr_.DrawPanels(visParams_);
		userInterfaceMgr_.EndFrame();

		// 4. render ImGui on top of the 3D scene
		userInterfaceMgr_.RenderDrawData();

		glfwSwapBuffers(window_);
	}
}

void VisualizationMgr::SetVisualMailbox(Sync::VisualSnapMailbox* visualSnapMailbox)
{
	visualSnapMailbox_ = visualSnapMailbox;
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
		SPDLOG_WARN("Difference between x/y scale when determining dpi");
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
		userInputMgr_.OnKey(visParams_.camera, key, scancode, action, mods);
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
	}
}

void VisualizationMgr::SyncVisSimState()
{
	if (!demoActive_) { return; }

	if (!demoPrimed_) 
	{
		position_[back_].resize(demoCount_);
		scalarData_[back_].resize(demoCount_);

		std::mt19937 rng{ 12345678 }; // deterministic seed
		std::normal_distribution<float> N(0.0f, 0.5f);

		for (std::int64_t i = 0; i < demoCount_; i++)
		{
			glm::vec3 p{ N(rng), N(rng), N(rng) };
			position_[back_][i] = p;
			scalarData_[back_][i] = glm::length(p);
		}

		SwapVisBuffers();

		demoPrimed_ = true;
	}

	view_.position = std::span<const glm::vec3>(position_[front_].data(), position_[front_].size());
	view_.scalarData = std::span<const float>(scalarData_[front_].data(), scalarData_[front_].size());
	view_.scalarMode = scalarMode_;
	view_.N_particles = position_[front_].size();
	view_.changed = true;

	renderingMgr_.UpdateParticles(view_);
}

void VisualizationMgr::SwapVisBuffers() { std::swap(front_, back_); }
