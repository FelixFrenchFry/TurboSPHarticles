// user_interface.cpp

// external
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <implot.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <spdlog/spdlog.h>

// internal
#include "Source/Base/helpers.h"
#include "Source/Vis/Assets/iosevka_compressed.h"
#include "user_interface.h"



using namespace Inter;

UserInterfaceMgr::UserInterfaceMgr(Sim::SimParamsMgr& simParamsMgr,
								   Vis::VisParamsMgr& visParamsMgr)
	: simParamsMgr_{ simParamsMgr }
	, visParamsMgr_{ visParamsMgr }
{
	// handled by Init()
}

void UserInterfaceMgr::Init(GLFWwindow* window,
							const Sim::SimParams& simParams,
							const char* glslVersion,
							const char* iniFilename,
							float dpiScale)
{
	if (initialized_) { return; }

	if (!window || !glslVersion)
	{
		SPDLOG_ERROR("Failed to init, window or glslVersion is null");
		return;
	}

	window_ = window;
	iniFilename_ = iniFilename ? iniFilename : "";

	// create ImGui and ImPlot contexts
	CreateContexts(iniFilename_.c_str());

	// overwrite ImGui backends
	ImGui_ImplGlfw_InitForOpenGL(window_, false);
	ImGui_ImplOpenGL3_Init(glslVersion);

	// initial param values
	temp_ =
	{
		.taitExponent = simParams.taitExponent,
		.stiffness = simParams.stiffness,
		.viscosityF = simParams.viscosityF,
		.viscosityB = simParams.viscosityB,
		.gravity = simParams.gravity.y
	};

	// style & flags
	SetupStyle();
	RebuildFonts(dpiScale);

	initialized_ = true;
}

void UserInterfaceMgr::Shutdown()
{
	if (!initialized_) { return; }

	// backends first
	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();

	// ImGui and ImPlot second
	DestroyContexts();

	window_ = nullptr;
	initialized_ = false;
}

void UserInterfaceMgr::BeginFrame()
{
	if (!initialized_) { return; }

	ImGui_ImplOpenGL3_NewFrame();
	ImGui_ImplGlfw_NewFrame();
	ImGui::NewFrame();

	// optionally create root dockspace
	if (dockingEnabled_) BeginDockspace();
}

void UserInterfaceMgr::BuildUserInterfaceSections(Sim::SimulationMgr& simulationMgr,
												  const Sim::SimParams& simParams,
												  const Vis::VisParams& visParams)
{
	if (!initialized_) { return; }

	ImGui::Begin("Settings");

	if (ImGui::CollapsingHeader("Scene Management", ImGuiTreeNodeFlags_DefaultOpen))
	{
		BuildSection_SceneManagement(simulationMgr, simParams, visParams);
	}

	if (ImGui::CollapsingHeader("Simulation Control", ImGuiTreeNodeFlags_DefaultOpen))
	{
		BuildSection_SimulationControl(simulationMgr, simParams, visParams);
	}

	if (ImGui::CollapsingHeader("Parameters"))
	{
		BuildSection_Parameters(simulationMgr, simParams, visParams);
	}

	// show FPS
	ImGui::Text("FPS: %.0f", ImGui::GetIO().Framerate);

	// placeholder controls
	ImGui::SeparatorText("(Placeholder)");
	ImGui::BeginDisabled(true);
	{
		float dummyRadius = 0.01f;
		int   dummyLod = 0;
		ImGui::SliderFloat("Sphere radius", &dummyRadius, 0.001f, 0.05f, "%.3f");
		ImGui::SliderInt("Sphere LOD", &dummyLod, 0, 2);
		static int scalar = 0; // 0=None, 1=Vel, 2=Density, 3=Pressure
		ImGui::Combo("Scalar", &scalar, "None\0Velocity\0Density\0Pressure\0");
	}
	ImGui::EndDisabled();

	// hints about input capture
	ImGui::SeparatorText("Capture");
	ImGuiIO& io = ImGui::GetIO();
	ImGui::TextDisabled("mouse=%d keyboard=%d",
		io.WantCaptureMouse ? 1 : 0,
		io.WantCaptureKeyboard ? 1 : 0);
	ImGui::End();
}

void UserInterfaceMgr::EndFrame()
{
	if (!initialized_) { return; }

	ImGui::End();
}

// TODO: call AFTER scene rendering
void UserInterfaceMgr::RenderDrawData()
{
	if (!initialized_) { return; }

	ImGui::Render();
	ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

// ===== callback instance handlers =====

bool UserInterfaceMgr::OnKey(int key, int scancode, int action, int mods)
{
	if (!initialized_) { return false; }

	ImGui_ImplGlfw_KeyCallback(window_, key, scancode, action, mods);

	return ImGui::GetIO().WantCaptureKeyboard;
}

bool UserInterfaceMgr::OnMouseButton(int button, int action, int mods)
{
	if (!initialized_) { return false; }

	ImGui_ImplGlfw_MouseButtonCallback(window_, button, action, mods);

	return ImGui::GetIO().WantCaptureMouse;
}

bool UserInterfaceMgr::OnCursorPos(double xPos, double yPos)
{
	if (!initialized_) { return false; }

	ImGui_ImplGlfw_CursorPosCallback(window_, xPos, yPos);

	return ImGui::GetIO().WantCaptureMouse;
}

bool UserInterfaceMgr::OnScroll(double xOff, double yOff)
{
	if (!initialized_) { return false; }

	ImGui_ImplGlfw_ScrollCallback(window_, xOff, yOff);

	return ImGui::GetIO().WantCaptureMouse;
}

bool UserInterfaceMgr::OnChar(unsigned int codepoint)
{
	if (!initialized_) { return false; }

	ImGui_ImplGlfw_CharCallback(window_, codepoint);

	return ImGui::GetIO().WantCaptureKeyboard;
}

// TODO: is this needed?
void UserInterfaceMgr::OnDPIScaleChanged(float dpiScale)
{
	if (!initialized_) return;

	// TODO: figure out how to store dpi properties
	// dpiX_ = xScale;
	// dpiY_ = yScale;

	// rebuild fonts for the new dpi for higher quality
	RebuildFonts(dpiScale);
}

// ===== sections =====

void UserInterfaceMgr::BuildSection_SceneManagement(
	Sim::SimulationMgr& simulationMgr,
	const Sim::SimParams& simParams,
	const Vis::VisParams& visParams)
{
	ImGui::Spacing();

	// temporary flags
	bool cached = false;

	// TODO: select scenario preset via dropdown?

	ImGui::PushStyleColor(ImGuiCol_Button, RGBA_negative_);
	if (simulationMgr.simStep_ == 0) { ImGui::BeginDisabled(); cached = true; }
	if (ImGui::Button(" RESET ", { 185, 40 }))
	{
		simulationMgr.ReBuildInitialSimState();
		simParamsMgr_.Set_SimRunning(false);
	}
	if (cached) { ImGui::EndDisabled(); cached = false; }
	ImGui::PopStyleColor(1);

	if (ImGui::TreeNode("scene settings"))
	{
		ImGui::Spacing();
		ImGui::SeparatorText("(Placeholder)");

		ImGui::TreePop();
	}
	ImGui::Spacing();
}

void UserInterfaceMgr::BuildSection_SimulationControl(
	const Sim::SimulationMgr& simulationMgr,
	const Sim::SimParams& simParams,
	const Vis::VisParams& visParams)
{
	ImGui::Spacing(); ImGui::Spacing();

	ImGui::SameLine();
	if (simParams.simRunning)
	{
		//ImGui::SetCursorPosX(GetTextAlignment_Right("", 10));
		ImGui::PushStyleColor(ImGuiCol_Button, RGBA_negative_);
		if (ImGui::Button("PAUSE SIMULATION", { 240, 40 }))
		{
			simParamsMgr_.Set_SimRunning(false);
		}
	}
	else
	{
		//ImGui::SetCursorPosX(GetTextAlignment_Right("", 10));
		ImGui::PushStyleColor(ImGuiCol_Button, RGBA_positive_);
		if (ImGui::Button("RESUME SIMULATION", { 240, 40 }))
		{
			simParamsMgr_.Set_SimRunning(true);
		}
	}
	ImGui::PopStyleColor(1);
	ImGui::SameLine();
	if (ImGui::Button("STEP +1", { 134, 40 }))
	{
		simParamsMgr_.Set_SimRunNextStep(true);
	}
	ImGui::Spacing();

	ImGui::Text("step count:");
	ImGui::SameLine();
	snprintf(stringBuffer, sizeof(stringBuffer), "%d", simulationMgr.simStep_);
	//ImGui::SetCursorPosX(GetTextAlignment_Right(stringBuffer, 230));
	ImGui::Text(stringBuffer);
	ImGui::Spacing();

	/*
	// update displayed FPS value in specified time intervals
	ImGui::SameLine();
	refreshTimeDifference = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - lastFrameRateRefreshTime).count();
	if (refreshTimeDifference > 1000.0f * frameRateRefreshInterval)
	{
		frameRate = ImGui::GetIO().Framerate;
		lastFrameRateRefreshTime = std::chrono::high_resolution_clock::now();
	}
	if (!parameterHandler->simulationPaused) { snprintf(stringBuffer, sizeof(stringBuffer), "(%.1f/sec)", frameRate); }
	else { snprintf(stringBuffer, sizeof(stringBuffer), "(----/sec)"); }
	ImGui::SetCursorPosX(GetTextAlignment_Right(stringBuffer, 383));
	ImGui::Text(stringBuffer);
	if (fluidParticles->particleCount == 0) { ImGui::EndDisabled(); }
	ImGui::Spacing();
	*/
}

void UserInterfaceMgr::BuildSection_Parameters(
	const Sim::SimulationMgr& simulationMgr,
	const Sim::SimParams& simParams,
	const Vis::VisParams& visParams)
{
	ImGui::Spacing();

	BuildParamElement_TaitExponent(simParams);
	BuildParamElement_Stiffness(simParams);
	BuildParamElement_ViscosityF(simParams);
	BuildParamElement_ViscosityB(simParams);
	BuildParamElement_Gravity(simParams);

	ImGui::Spacing();

	// save or discard parameter changes buttons
	const bool cached = unsavedParamChanges_;
	if (!cached) { ImGui::BeginDisabled(); }
	ImGui::PushStyleColor(ImGuiCol_Button, RGBA_positive_);
	if (ImGui::Button(" APPLY CHANGES ", { 185, 40 }))
	{
		ApplyParamChanges();
	}
	ImGui::SameLine();
	ImGui::PopStyleColor(1);
	ImGui::PushStyleColor(ImGuiCol_Button, RGBA_negative_);
	if (ImGui::Button(" RESET CHANGES ", { 185, 40 }))
	{
		ResetParamChanges(simParams);
	}
	ImGui::PopStyleColor(1);
	ImGui::Spacing();
	if (!cached) { ImGui::EndDisabled(); }

	// live editing checkbox and tooltip
	if (ImGui::Checkbox("##checkbox_liveParamEditing", &liveParamEditing_)) {}
	ImGui::SameLine();
	ImGui::Button(" ? ");
	if (ImGui::IsItemHovered())
	{
		ImGui::BeginTooltip();
		ImGui::TextColored(RGBA_tooltip_,
		"With this enabled, any changes to the\n"
			"parameters are applied immediately.\n"
			"There is no option to APPLY or RESET.");
		ImGui::EndTooltip();
	}
	ImGui::SameLine();
	ImGui::Text(" enable live editing");
	ImGui::Spacing();
}

void UserInterfaceMgr::BuildParamElement_TaitExponent(const Sim::SimParams& simParams)
{
	const bool cached = changed_.taitExponent;
	if (cached) { ImGui::PushStyleColor(ImGuiCol_Text, RGBA_unsaved_); }
	ImGui::SetNextItemWidth(170);
	if (ImGui::InputFloat(" tait exponent##input_taitExponent",
		&temp_.taitExponent, mags_.taitExponent[0], mags_.taitExponent[1], "%.1f"))
	{
		const bool changed = !Base::ApproxEqu(temp_.taitExponent, simParams.taitExponent);

		if (changed && liveParamEditing_) // apply param changes immediately
		{
			simParamsMgr_.Set_TaitExponent(temp_.taitExponent);
		}

		// mark unsaved changes when not live editing
		changed_.taitExponent = changed && !liveParamEditing_;
		unsavedParamChanges_ = changed && !liveParamEditing_;
	}
	if (cached) { ImGui::PopStyleColor(); }
}

void UserInterfaceMgr::BuildParamElement_Stiffness(const Sim::SimParams& simParams)
{
	const bool cached = changed_.stiffness;
	if (cached) { ImGui::PushStyleColor(ImGuiCol_Text, RGBA_unsaved_); }
	ImGui::SetNextItemWidth(170);
	if (ImGui::InputFloat(" stiffness##input_stiffness",
		&temp_.stiffness, mags_.stiffness[0], mags_.stiffness[1], "%.0f"))
	{
		const bool changed = !Base::ApproxEqu(temp_.stiffness, simParams.stiffness);

		if (changed && liveParamEditing_) // apply param changes immediately
		{
			simParamsMgr_.Set_Stiffness(temp_.stiffness);
		}

		// mark unsaved changes when not live editing
		changed_.stiffness = changed && !liveParamEditing_;
		unsavedParamChanges_ = changed && !liveParamEditing_;
	}
	if (cached) { ImGui::PopStyleColor(); }
}

void UserInterfaceMgr::BuildParamElement_ViscosityF(const Sim::SimParams& simParams)
{
	const bool cached = changed_.viscosityF;
	if (cached) { ImGui::PushStyleColor(ImGuiCol_Text, RGBA_unsaved_); }
	ImGui::SetNextItemWidth(170);
	if (ImGui::InputFloat(" fluid visco##input_viscosityF",
		&temp_.viscosityF, mags_.viscosityF[0], mags_.viscosityF[1], "%.6f"))
	{
		const bool changed = !Base::ApproxEqu(temp_.viscosityF, simParams.viscosityF);

		if (changed && liveParamEditing_) // apply param changes immediately
		{
			simParamsMgr_.Set_Viscosity(temp_.viscosityF, Base::ParticleType::Fluid);
		}

		// mark unsaved changes when not live editing
		changed_.viscosityF = changed && !liveParamEditing_;
		unsavedParamChanges_ = changed && !liveParamEditing_;
	}
	if (cached) { ImGui::PopStyleColor(); }
}

void UserInterfaceMgr::BuildParamElement_ViscosityB(const Sim::SimParams& simParams)
{
	const bool cached = changed_.viscosityB;
	if (cached) { ImGui::PushStyleColor(ImGuiCol_Text, RGBA_unsaved_); }
	ImGui::SetNextItemWidth(170);
	if (ImGui::InputFloat(" bound visco##input_viscosityB",
		&temp_.viscosityB, mags_.viscosityB[0], mags_.viscosityB[1], "%.6f"))
	{
		const bool changed = !Base::ApproxEqu(temp_.viscosityB, simParams.viscosityB);

		if (changed && liveParamEditing_) // apply param changes immediately
		{
			simParamsMgr_.Set_Viscosity(temp_.viscosityB, Base::ParticleType::Boundary);
		}

		// mark unsaved changes when not live editing
		changed_.viscosityB = changed && !liveParamEditing_;
		unsavedParamChanges_ = changed && !liveParamEditing_;
	}
	if (cached) { ImGui::PopStyleColor(); }
}

void UserInterfaceMgr::BuildParamElement_Gravity(const Sim::SimParams& simParams)
{
	const bool cached = changed_.gravity;
	if (cached) { ImGui::PushStyleColor(ImGuiCol_Text, RGBA_unsaved_); }
	ImGui::SetNextItemWidth(170);
	if (ImGui::InputFloat(" gravity##input_gravity",
		&temp_.gravity, mags_.gravity[0], mags_.gravity[1], "%.2f"))
	{
		const bool changed = !Base::ApproxEqu(temp_.gravity, simParams.gravity.y);

		if (changed && liveParamEditing_) // apply param changes immediately
		{
			simParamsMgr_.Set_Gravity(glm::vec3{ 0.f, temp_.gravity, 0.f });
		}

		// mark unsaved changes when not live editing
		changed_.gravity = changed && !liveParamEditing_;
		unsavedParamChanges_ = changed && !liveParamEditing_;
	}
	if (cached) { ImGui::PopStyleColor(); }
}

// ===== helpers =====

void UserInterfaceMgr::SetDockingEnabled(bool enabled)
{
	ImGuiIO& io = ImGui::GetIO();
	if (enabled) io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
	else         io.ConfigFlags &= ~ImGuiConfigFlags_DockingEnable;

	dockingEnabled_ = enabled;
}

void UserInterfaceMgr::CreateContexts(const char* iniFilename)
{
	// ImGui & ImPlot context
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImPlot::CreateContext();

	ImGuiIO& io = ImGui::GetIO();
	io.IniFilename = iniFilename && *iniFilename ? iniFilename : nullptr;
	io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
	io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
}

void UserInterfaceMgr::DestroyContexts()
{
	ImPlot::DestroyContext();
	ImGui::DestroyContext();
}

void UserInterfaceMgr::SetupStyle()
{
	ImGui::StyleColorsDark();

	// TODO: custom style tweaks
	ImGuiStyle& style = ImGui::GetStyle();
	style.WindowPadding = ImVec2(4, 4);
	style.FramePadding = ImVec2(4, 2);
	style.ItemSpacing = ImVec2(4, 3);
	style.ScrollbarSize = 14;
}

void UserInterfaceMgr::RebuildFonts(float dpiScale)
{
	ImGuiIO& io = ImGui::GetIO();

	const float baseSize = 24.0f;
	const float pixelSize = baseSize * (dpiScale > 0.0f ? dpiScale : 1.0f);

	io.Fonts->Clear();
	ImFont* font = nullptr;
	font = io.Fonts->AddFontFromMemoryCompressedTTF(
		iosevka_compressed_data,
		iosevka_compressed_size,
		pixelSize
	);
	if (!font)
	{
		SPDLOG_ERROR("Failed to load custom compressed font from header file");
		io.Fonts->AddFontDefault();
	}

	ImGui_ImplOpenGL3_DestroyFontsTexture();
	ImGui_ImplOpenGL3_CreateFontsTexture();
}

void UserInterfaceMgr::ApplyParamChanges()
{
	if (!unsavedParamChanges_) { return; }

	if (changed_.taitExponent)
	{
		simParamsMgr_.Set_TaitExponent(temp_.taitExponent);
		changed_.taitExponent = false;
	}
	if (changed_.stiffness)
	{
		simParamsMgr_.Set_Stiffness(temp_.stiffness);
		changed_.stiffness = false;
	}
	if (changed_.viscosityF)
	{
		simParamsMgr_.Set_Viscosity(temp_.viscosityF, Base::ParticleType::Fluid);
		changed_.viscosityF = false;
	}
	if (changed_.viscosityB)
	{
		simParamsMgr_.Set_Viscosity(temp_.viscosityB, Base::ParticleType::Boundary);
		changed_.viscosityB = false;
	}
	if (changed_.gravity)
	{
		simParamsMgr_.Set_Gravity(glm::vec3{ 0.0f, temp_.gravity, 0.0f });
		changed_.gravity = false;
	}

	unsavedParamChanges_ = false;
}

void UserInterfaceMgr::ResetParamChanges(const Sim::SimParams& simParams)
{
	if (!unsavedParamChanges_) { return; }

	if (changed_.taitExponent)
	{
		temp_.taitExponent = simParams.taitExponent;
		changed_.taitExponent = false;
	}
	if (changed_.stiffness)
	{
		temp_.stiffness = simParams.stiffness;
		changed_.stiffness = false;
	}
	if (changed_.viscosityF)
	{
		temp_.viscosityF = simParams.viscosityF;
		changed_.viscosityF = false;
	}
	if (changed_.viscosityB)
	{
		temp_.viscosityB = simParams.viscosityB;
		changed_.viscosityB = false;
	}
	if (changed_.gravity)
	{
		temp_.gravity = simParams.gravity.y;
		changed_.gravity = false;
	}

	unsavedParamChanges_ = false;
}

// ===== dockspace =====

void UserInterfaceMgr::BeginDockspace()
{
	// full-screen invisible dummy window as dockspace host
	ImGuiWindowFlags windowFlags =
		ImGuiWindowFlags_NoDocking |
		ImGuiWindowFlags_NoTitleBar |
		ImGuiWindowFlags_NoCollapse |
		ImGuiWindowFlags_NoResize |
		ImGuiWindowFlags_NoMove |
		ImGuiWindowFlags_NoBringToFrontOnFocus |
		ImGuiWindowFlags_NoNavFocus |
		ImGuiWindowFlags_NoBackground;

	const ImGuiViewport* vp = ImGui::GetMainViewport();
	ImGui::SetNextWindowPos(vp->Pos);
	ImGui::SetNextWindowSize(vp->Size);
	ImGui::SetNextWindowViewport(vp->ID);

	ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
	ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);

	ImGui::Begin("##DockSpaceRoot", nullptr, windowFlags);
	ImGui::PopStyleVar(2);

	// central dockspace node that other windows can dock into
	ImGuiID dockId = ImGui::GetID("MainDockSpace");
	ImGui::DockSpace(dockId, ImVec2(0.0f, 0.0f),
		ImGuiDockNodeFlags_PassthruCentralNode);
}
