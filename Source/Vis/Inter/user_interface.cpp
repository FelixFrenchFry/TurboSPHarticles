// user_interface.cpp

// external
#include <cstdarg>
#include <cstdio>
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
												  const Bench::BenchmarksMgr& benchmarksMgr,
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

	if (ImGui::CollapsingHeader("Simulation Parameters"))
	{
		BuildSection_SimulationParameters(simulationMgr, simParams, visParams);
	}

	if (ImGui::CollapsingHeader("Simulation Statistics"))
	{
		// BuildSection_SimulationStatistics(simulationMgr, simParams, visParams);
	}

	if (ImGui::CollapsingHeader("Performance Benchmarks"))
	{
		std::vector<Bench::BenchOpGroupDef> opGroups;
		simulationMgr.GetBenchOpGroups(opGroups);
		BuildSection_PerformanceBenchmarks(simulationMgr, benchmarksMgr, opGroups, simParams, visParams);
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

void UserInterfaceMgr::BuildSection_SimulationParameters(
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

void UserInterfaceMgr::BuildSection_SimulationStatistics(
	const Sim::SimulationMgr& simulationMgr, 
	const Sim::SimParams& simParams, 
	const Vis::VisParams& visParams)
{
	// TODO
}

// TODO: deal with threading, which data to show, update frequency, syncing, etc...
void UserInterfaceMgr::BuildSection_PerformanceBenchmarks(
	const Sim::SimulationMgr& simulationMgr,
	const Bench::BenchmarksMgr& benchmarksMgr,
	const std::vector<Bench::BenchOpGroupDef>& opGroups,
	const Sim::SimParams& simParams,
	const Vis::VisParams& visParams)
{
	using BTV = BenchTableView;

	ImGui::Spacing();

	// tab bar for selecting the table view ------------------------------------

	if (ImGui::BeginTabBar("tabBar_bench"))
	{
		// helper function for the tabs
		auto MakeTab = [&](BTV view, const char* label) -> void
		{
			if (ImGui::BeginTabItem(label))
			{
				benchTableView_ = view;
				ImGui::EndTabItem();
			}
		};

		MakeTab(BTV::Latest, "Current");
		MakeTab(BTV::Avg_All, "Avg all");
		MakeTab(BTV::Avg_Used, "Avg used");
		MakeTab(BTV::Min, "Min");
		MakeTab(BTV::Max, "Max");
		ImGui::EndTabBar();
	}

	// no percentage column for min/max views
	const bool showPercent = !(benchTableView_ == BTV::Min ||
							   benchTableView_ == BTV::Max);

	// per-group timing and totals ---------------------------------------------

	// timing per group (row)
	std::vector<uint64_t> groupTimings(opGroups.size(), 0);

	// sum of all groups except the one for BenchOp::Total_Sim
	// (used as denominator for percentages)
	uint64_t colTime = 0;
	int totalSimRowIdx = -1;

	for (uint32_t i = 0; i < opGroups.size(); i++)
	{
		const auto& group = opGroups[i];

		// pick representative timing for each group 
		const uint64_t t = ReadOpGroup_Min(benchmarksMgr, group);
		groupTimings[i] = t;

		//SPDLOG_ERROR("time: {}", t);

		// detect BenchOp::Total_Sim row to exclude from percentage calculations
		if (group.ops.size() == 1 && group.ops[0] == Bench::BenchOp::Total_Sim)
		{
			totalSimRowIdx = static_cast<int>(i);
		}
		else
		{
			colTime += t;
		}
	}

	// benchmark table ---------------------------------------------------------

	const int numCols = showPercent ? 3 : 2;

	if (ImGui::BeginTable("table_bench", numCols,
		ImGuiTableFlags_RowBg |
		ImGuiTableFlags_BordersH |
		ImGuiTableFlags_Resizable |
		ImGuiTableFlags_SizingStretchProp))
	{
		// column headers
		ImGui::TableSetupColumn("operation group",
			ImGuiTableColumnFlags_WidthStretch, 0.65f);
		ImGui::TableSetupColumn(UnitLabel(benchTimeUnit_),
			ImGuiTableColumnFlags_WidthStretch, 0.20f);
		if (showPercent)
		{
			ImGui::TableSetupColumn("%",
				ImGuiTableColumnFlags_WidthStretch, 0.15f);
		}
		ImGui::TableHeadersRow();

		// helper function for drawing each row
		auto DrawOpGroupRow = [&](uint32_t rowIdx, bool isTotalSim) -> void
		{
			const auto& group = opGroups[rowIdx];
			const uint64_t t = groupTimings[rowIdx];
			const double t_unit = ConvertFromNs(benchTimeUnit_, t);

			ImGui::TableNextRow();

			// highlight BenchOp::Total_Sim row
			if (isTotalSim)
			{
				ImGui::TableSetBgColor(ImGuiTableBgTarget_RowBg0, IM_COL32(200, 50, 50, 255));
				ImGui::TableSetBgColor(ImGuiTableBgTarget_RowBg1, IM_COL32(200, 50, 50, 255));
			}

			// column 0 with aliases
			// (expandable to show full names vertically)
			ImGui::TableSetColumnIndex(0);
			const std::string alias = MakeGroupAlias(group);

			// stable ID for each row
			ImGui::PushID(rowIdx);
			if (isTotalSim)
			{
				// show as a regular row (no tree)
				ImGui::TextUnformatted("total simulation time");
			}
			else
			{
				constexpr ImGuiTreeNodeFlags flags =
					ImGuiTreeNodeFlags_SpanFullWidth |
					ImGuiTreeNodeFlags_AllowItemOverlap |
					ImGuiTreeNodeFlags_FramePadding;

				if (ImGui::TreeNodeEx("##group", flags, "%s", alias.c_str()))
				{
					// make the unfolded list start closer to the left
					ImGui::PushStyleVar(ImGuiStyleVar_IndentSpacing, 50.0f);

					// show full names (one per line)
					for (uint32_t k = 0; k < group.ops.size(); k++)
					{
						const auto op = group.ops[k];
						const auto& def = Bench::Definitions[Bench::GetIdx(op)];
						ImGui::BulletText("%s", def.name.data());
					}

					ImGui::PopStyleVar();
					ImGui::TreePop();
				}
			}
			ImGui::PopID();

			// column 1 with timing in selected unit
			ImGui::TableSetColumnIndex(1);
			TableTextRightAligned("%.1f", t_unit);

			// column 2 with percentages of total sim time
			if (showPercent)
			{
				ImGui::TableSetColumnIndex(2);
				if (!isTotalSim && colTime > 0)
				{
					const float pct = 100.0f * (static_cast<float>(t)
									/ static_cast<float>(colTime));
					const float frac = pct / 100.0f;

					char pctLabel[32];
					std::snprintf(pctLabel, sizeof(pctLabel), "%.1f%%", pct);

					ImGui::PushStyleColor(ImGuiCol_PlotHistogram, IM_COL32(230, 150, 0, 255));
					ImGui::ProgressBar(frac, ImVec2(-FLT_MIN, 0.0f), pctLabel);
					ImGui::PopStyleColor(1);
				}
				else
				{
					ImGui::Text("---");
				}
			}
		};

		// draw all regular rows first
		for (uint32_t i = 0; i < opGroups.size(); i++)
		{
			if (static_cast<int>(i) == totalSimRowIdx) { continue; }

			DrawOpGroupRow(i, false);
		}

		// draw BenchOp::Total_Sim row last (if present)
		if (totalSimRowIdx >= 0)
		{
			DrawOpGroupRow(totalSimRowIdx, true);
		}

		ImGui::EndTable();
	}

	// TODO: sparkline?
	// sparkline for BenchOp::Total_Sim ----------------------------------------

	// display settings --------------------------------------------------------

	if (ImGui::TreeNodeEx("measurement settings"))
	{
		// simple unit pickers (mutually exclusive)
		bool ns = (benchTimeUnit_ == TimeUnit::ns);
		bool us = (benchTimeUnit_ == TimeUnit::us);
		bool ms = (benchTimeUnit_ == TimeUnit::ms);

		ImGui::Text("time unit: ");
		ImGui::SameLine();
		if (ImGui::Checkbox("ms ", &ms)) { benchTimeUnit_ = TimeUnit::ms; ns = us = false; }
		ImGui::SameLine();
		if (ImGui::Checkbox("us ", &us)) { benchTimeUnit_ = TimeUnit::us; ns = ms = false; }
		ImGui::SameLine();
		if (ImGui::Checkbox("ns", &ns)) { benchTimeUnit_ = TimeUnit::ns; us = ms = false; }
		ImGui::Spacing();

		// number of steps shown in sparkline
		ImGui::SetNextItemWidth(120.0f);
		ImGui::SliderInt("history", &sparklineContextLength_, 32, 1024);

		ImGui::TreePop();
	}
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

inline double UserInterfaceMgr::ConvertFromNs(TimeUnit unit, uint64_t ns)
{
	switch (unit)
	{
	case TimeUnit::ms: { return static_cast<double>(ns) / 1'000'000.0; }
	case TimeUnit::us: { return static_cast<double>(ns) / 1'000.0; }
	case TimeUnit::ns: { return static_cast<double>(ns); }
	}

	SPDLOG_WARN("Invalid TimeUnit enum value");
	return static_cast<double>(ns);
}

inline const char* UserInterfaceMgr::UnitLabel(TimeUnit unit)
{
	switch (unit)
	{
	case TimeUnit::ns: { return "ns"; }
	case TimeUnit::us: { return "us"; }
	case TimeUnit::ms: { return "ms"; }
	}

	SPDLOG_WARN("Invalid TimeUnit enum value");
	return "";
}

inline uint64_t UserInterfaceMgr::ReadOpTiming(const Bench::BenchmarksMgr& benchmarksMgr, Bench::BenchOp op)
{
	switch (benchTableView_)
	{
	case BenchTableView::Latest: { return benchmarksMgr.GetBench_Latest(op); }
	case BenchTableView::Avg_All: { return benchmarksMgr.GetBench_Avg_OverAll(op); }
	case BenchTableView::Avg_Used: { return benchmarksMgr.GetBench_Avg_WhenUsed(op); }
	case BenchTableView::Min: { return benchmarksMgr.GetBench_Min(op); }
	case BenchTableView::Max: { return benchmarksMgr.GetBench_Max(op); }
	}

	SPDLOG_WARN("Invalid BenchTableView enum value");
	return 0;
}

inline uint64_t UserInterfaceMgr::ReadOpGroup_Min(const Bench::BenchmarksMgr& benchmarksMgr,
												  const Bench::BenchOpGroupDef& opGroup)
{
	uint64_t res = std::numeric_limits<uint64_t>::max();

	// pick one of the operation timings as a representative
	// (they should all be the same value, but the groups are variably-sized)
	for (const auto& op : opGroup.ops)
	{
		res = std::min(res, ReadOpTiming(benchmarksMgr, op));
	}

	return res;
}

inline std::string UserInterfaceMgr::MakeGroupAlias(const Bench::BenchOpGroupDef& opGroup)
{
	std::string res;

	// build comma-separated list of operation aliases
	for (uint32_t i = 0; i < opGroup.ops.size(); i++)
	{
		if (i > 0) { res += ", "; }
		res += Bench::Definitions[Bench::GetIdx(opGroup.ops[i])].alias;
	}

	return res;
}

void UserInterfaceMgr::TableTextRightAligned(const char* fmt, ...)
{
	char buf[256];

	va_list args;
	va_start(args, fmt);
	int n = std::vsnprintf(buf, sizeof(buf), fmt, args);
	va_end(args);

	// compute right-aligned cursor X within the current table cell
	const float avail = ImGui::GetContentRegionAvail().x;
	const float textWidth = ImGui::CalcTextSize(buf).x;
	const float x = ImGui::GetCursorPosX() + std::max(0.0f, avail - textWidth - 2.0f);

	ImGui::SetCursorPosX(x);
	ImGui::TextUnformatted(buf);
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
