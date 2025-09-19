// user_interface.cpp

// external
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <implot.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <spdlog/spdlog.h>

// internal
#include "../Assets/iosevka_compressed.h"
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

void UserInterfaceMgr::DrawPanels(const Vis::VisParams& visParams)
{
	if (!initialized_) { return; }

	if (ImGui::Begin("Settings"))
	{
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
	}
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

	const float baseSize = 22.0f;
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
	ImGui::DockSpace(dockId, ImVec2(0.0f, 0.0f), ImGuiDockNodeFlags_PassthruCentralNode);
}
