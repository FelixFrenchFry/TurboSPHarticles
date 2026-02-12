// user_interface.cpp

// external
#include <algorithm>
#include <cmath>
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
		.timeStep = simParams.timeStep,
		.viscosityF = simParams.viscosityF,
		.viscosityB = simParams.viscosityB,
		.pressGamma = simParams.pressGamma,
		.jacobiOmega = simParams.jacobiOmega,
		.densityDevTarget = simParams.densityDevTarget,
		.minSolverIters = static_cast<int>(simParams.minSolverIters),
		.maxSolverIters = static_cast<int>(simParams.maxSolverIters),
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

void UserInterfaceMgr::BuildUserInterfaceSections(
	Sim::SimulationMgr& simulationMgr,
	const Sync::BenchSnapshot* benchSnap,
	const Sync::StatSnapshot* statSnap,
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
		BuildSection_SimulationStatistics(simulationMgr, statSnap, simParams, visParams);
	}

	if (ImGui::CollapsingHeader("Performance Benchmarks"))
	{
		std::vector<Bench::BenchOpGroupDef> opGroups;
		simulationMgr.GetBenchOpGroups(opGroups);
		BuildSection_PerformanceBenchmarks(simulationMgr, benchSnap, opGroups, simParams, visParams);
	}

	// show FPS
	ImGui::Text("FPS: %.0f", ImGui::GetIO().Framerate);

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

// --- [ callback instance handlers ] -------------------------------------------------------------

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

// --- [ sections ] -------------------------------------------------------------------------------

void UserInterfaceMgr::BuildSection_SceneManagement(
	Sim::SimulationMgr& simulationMgr,
	const Sim::SimParams& simParams,
	const Vis::VisParams& visParams)
{
	ImGui::Spacing();

	// temporary flags
	bool cached = false;

	// TODO: select scenario preset via dropdown?

	// simulation reset button
	ImGui::PushStyleColor(ImGuiCol_Button, RGBA_negative_);
	if (simulationMgr.simStep_ == 0) { ImGui::BeginDisabled(); cached = true; }
	if (ImGui::Button(" RESET ", { 185, 40 }))
	{
		simulationMgr.ReBuildInitialSimState();
		simParamsMgr_.Set_SimRunning(false);

		// TODO: reset statistics and benchmarks
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
	snprintf(stringBuffer, sizeof(stringBuffer), "%lld", simulationMgr.simStep_);
	ImGui::Text(stringBuffer);
	ImGui::Spacing();
}

void UserInterfaceMgr::BuildSection_SimulationParameters(
	const Sim::SimulationMgr& simulationMgr,
	const Sim::SimParams& simParams,
	const Vis::VisParams& visParams)
{
	ImGui::Spacing();

	BuildParamElement_TimeStep(simParams);
	BuildParamElement_ViscosityF(simParams);
	BuildParamElement_ViscosityB(simParams);
	BuildParamElement_PressGamma(simParams);
	BuildParamElement_JacobiOmega(simParams);
	BuildParamElement_DensityDevTarget(simParams);
	BuildParamElement_MinSolverIters(simParams);
	BuildParamElement_MaxSolverIters(simParams);
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
	const Sync::StatSnapshot* statSnap,
	const Sim::SimParams& simParams,
	const Vis::VisParams& visParams)
{
	using STV = StatTableView;

	ImGui::Spacing();

	if (!statSnap)
	{
		ImGui::TextDisabled("(No statistics data recorded yet)");
		ImGui::Spacing();
		return;
	}

	// --- [ tab bar for selecting the table view ] --------------------------------------------------

	if (ImGui::BeginTabBar("tabBar_stats"))
	{
		// helper function for the tabs
		auto MakeTab = [&](STV view, const char* label) -> void
		{
			if (ImGui::BeginTabItem(label))
			{
				statTableView_ = view;
				ImGui::EndTabItem();
			}
		};

		MakeTab(STV::Latest, "Latest");
		MakeTab(STV::Avg, "Avg");
		MakeTab(STV::Min, "Min");
		MakeTab(STV::Max, "Max");
		ImGui::EndTabBar();
	}

	// helper function for the per-group 2-column tables
	auto DrawStatGroupTable = [&](const char* tableIdSuffix,
	std::initializer_list<Stats::StatMetric> metrics)
	{
		// 2 columns (metric name | value)
		char tableId[64];
		std::snprintf(tableId, sizeof(tableId), "table_stats_%s", tableIdSuffix);

		if (ImGui::BeginTable(tableId, 2,
			ImGuiTableFlags_RowBg |
			ImGuiTableFlags_BordersH |
			ImGuiTableFlags_Resizable |
			ImGuiTableFlags_SizingStretchProp))
		{
			ImGui::TableSetupColumn("metric", 
				ImGuiTableColumnFlags_WidthStretch, 0.75f);
			ImGui::TableSetupColumn("value", 
				ImGuiTableColumnFlags_WidthStretch, 0.25f);

			for (auto metric : metrics)
			{
				const uint16_t i = Stats::GetIdx(metric);
				const auto& def = Stats::Definitions[i];
				const float val = ReadStatData(statSnap, metric);

				ImGui::TableNextRow();

				// column 0 (metric name)
				ImGui::TableSetColumnIndex(0);
				ImGui::TextUnformatted(def.name.data());

				// tooltip from metric definition
				if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayNormal))
				{
					const char* text;

					!def.tip.empty() ? text = def.tip.data()
									 : text = "(no description provided)";

					ImGui::BeginTooltip();
					ImGui::TextColored(RGBA_tooltip_, text);
					ImGui::EndTooltip();
				}

				// column 1 (formatted, right-aligned value)
				ImGui::TableSetColumnIndex(1);
				{
					char buf[64];
					FormatStatValue(def, buf, val, sizeof(buf));
					TableTextRightAligned("%s", buf);
				}
			}

			ImGui::EndTable();
		}
	};

	// --- [ per-group stat tables ] -----------------------------------------------------------------

	// particles and neighbors
	ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(255, 255, 255, 100));
	if (ImGui::TreeNodeEx("Particle Stats",
		ImGuiTreeNodeFlags_DefaultOpen))
	{
		ImGui::PopStyleColor(1);

		DrawStatGroupTable("table_particles",
		{
			Stats::StatMetric::NumF,
			Stats::StatMetric::NumB,
			Stats::StatMetric::NumNbrFF_Avg,
			Stats::StatMetric::NumNbrFF_P95,
			Stats::StatMetric::NumNbrFF_P99,
			Stats::StatMetric::NumNbrFF_Min,
			Stats::StatMetric::NumNbrFF_Max,
			Stats::StatMetric::NumNbrFB_Avg,
			Stats::StatMetric::NumNbrFB_P95,
			Stats::StatMetric::NumNbrFB_P99,
			Stats::StatMetric::NumNbrFB_Min,
			Stats::StatMetric::NumNbrFB_Max
		});

		ImGui::TreePop();
	}
	else { ImGui::PopStyleColor(1); }
	ImGui::Spacing();

	// density
	ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(255, 255, 255, 100));
	if (ImGui::TreeNodeEx("Density Stats",
		ImGuiTreeNodeFlags_DefaultOpen))
	{
		ImGui::PopStyleColor(1);

		DrawStatGroupTable("table_density",
		{
			Stats::StatMetric::DensRelError_Raw_Avg,
			Stats::StatMetric::DensRelError_Pos_Avg,
			Stats::StatMetric::DensRelError_Abs_Avg
		});

		ImGui::TreePop();
	}
	else { ImGui::PopStyleColor(1); }
	ImGui::Spacing();

	// pressure
	ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(255, 255, 255, 100));
	if (ImGui::TreeNodeEx("Pressure Stats",
		ImGuiTreeNodeFlags_DefaultOpen))
	{
		ImGui::PopStyleColor(1);

		DrawStatGroupTable("table_pressure",
			{
				Stats::StatMetric::NumSolverIters
			});

		ImGui::TreePop();
	}
	else { ImGui::PopStyleColor(1); }
	ImGui::Spacing();

	// velocity
	ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(255, 255, 255, 100));
	if (ImGui::TreeNodeEx("Velocity Stats",
		ImGuiTreeNodeFlags_DefaultOpen))
	{
		ImGui::PopStyleColor(1);

		DrawStatGroupTable("table_velocity",
		{
			Stats::StatMetric::CFL_Number,
			Stats::StatMetric::VelMag_Avg,
			Stats::StatMetric::VelMag_Min,
			Stats::StatMetric::VelMag_Max
		});

		ImGui::TreePop();
	}
	else { ImGui::PopStyleColor(1); }
	ImGui::Spacing();

	// settings
	if (ImGui::TreeNodeEx("Statistics Settings"))
	{
		ImGui::TextDisabled("(no settings yet)");

		ImGui::TreePop();
	}
	ImGui::Spacing();
}

// TODO: deal with threading, which data to show, update frequency, syncing, etc...
void UserInterfaceMgr::BuildSection_PerformanceBenchmarks(
	const Sim::SimulationMgr& simulationMgr,
	const Sync::BenchSnapshot* benchSnap,
	const std::vector<Bench::BenchOpGroupDef>& opGroups,
	const Sim::SimParams& simParams,
	const Vis::VisParams& visParams)
{
	using BTV = BenchTableView;

	ImGui::Spacing();

	if (!benchSnap)
	{
		ImGui::TextDisabled("(No benchmark data recorded yet)");
		ImGui::Spacing();
		return;
	}

	// --- [ tab bar for selecting the table view ] --------------------------------------------------

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

		MakeTab(BTV::Latest, "Latest");
		MakeTab(BTV::Avg_All, "Avg all");
		MakeTab(BTV::Avg_Used, "Avg used");
		MakeTab(BTV::Min, "Min");
		MakeTab(BTV::Max, "Max");
		ImGui::EndTabBar();
	}

	// no percentage column for min/max views
	const bool showPercent = !(benchTableView_ == BTV::Min ||
							   benchTableView_ == BTV::Max);

	// --- [ per-group timing and totals ] -----------------------------------------------------------

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
		const uint64_t t = ReadOpGroup_Min(benchSnap, group);
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

	// --- [ benchmark table ] -----------------------------------------------------------------------

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
		ImGui::TableSetupColumn(Bench::UnitLabel(benchTimeUnit_),
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
			const double t_unit = Bench::ConvertFromNs(benchTimeUnit_, t);

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
					ImGui::Unindent(10.0f); // shift to the left by arrow/bullet spacing

					// show full names (one per line)
					for (uint32_t k = 0; k < group.ops.size(); k++)
					{
						const auto op = group.ops[k];
						const auto& def = Bench::Definitions[Bench::GetIdx(op)];
						ImGui::Text("- %s", def.name.data());
					}

					ImGui::Indent(10.0f); // restore indent
					ImGui::TreePop();
				}
			}
			ImGui::PopID();

			// column 1 with timing in selected unit
			ImGui::TableSetColumnIndex(1);
			TableTextRightAligned("%.2f", t_unit);

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
	// sparkline for BenchOp::Total_Sim

	// --- [ display settings ] ----------------------------------------------------------------------

	if (ImGui::TreeNodeEx("Benchmark Settings"))
	{
		// simple unit pickers (mutually exclusive)
		bool s = (benchTimeUnit_ == Bench::TimeUnit::s);
		bool ms = (benchTimeUnit_ == Bench::TimeUnit::ms);
		bool us = (benchTimeUnit_ == Bench::TimeUnit::us);

		ImGui::Text("time unit: ");
		ImGui::SameLine();
		if (ImGui::Checkbox("s ", &s)) { benchTimeUnit_ = Bench::TimeUnit::s; ms = us = false; }
		ImGui::SameLine();
		if (ImGui::Checkbox("ms ", &ms)) { benchTimeUnit_ = Bench::TimeUnit::ms; s = us = false; }
		ImGui::SameLine();
		if (ImGui::Checkbox("us ", &us)) { benchTimeUnit_ = Bench::TimeUnit::us; s = ms = false; }
		ImGui::Spacing();

		// number of steps shown in sparkline
		ImGui::SetNextItemWidth(120.0f);
		ImGui::SliderInt("history", &sparklineContextLength_, 32, 1024);

		ImGui::TreePop();
	}
}

void UserInterfaceMgr::BuildParamElement_TimeStep(const Sim::SimParams& simParams)
{
	const bool cached = changed_.timeStep;
	if (cached) { ImGui::PushStyleColor(ImGuiCol_Text, RGBA_unsaved_); }
	ImGui::SetNextItemWidth(170);
	if (ImGui::InputFloat(" time step##input_timeStep",
		&temp_.timeStep, mags_.timeStep[0], mags_.timeStep[1], "%.6f"))
	{
		const bool changed = !Base::ApproxEqu(temp_.timeStep, simParams.timeStep);

		if (changed && liveParamEditing_) // apply param changes immediately
		{
			simParamsMgr_.Set_TimeStep(temp_.timeStep);
		}

		// mark unsaved changes when not live editing
		changed_.timeStep = changed && !liveParamEditing_;
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

void UserInterfaceMgr::BuildParamElement_PressGamma(const Sim::SimParams& simParams)
{
	const bool cached = changed_.pressGamma;
	if (cached) { ImGui::PushStyleColor(ImGuiCol_Text, RGBA_unsaved_); }
	ImGui::SetNextItemWidth(170);
	if (ImGui::InputFloat(" gamma press##input_pressGamma",
		&temp_.pressGamma, mags_.pressGamma[0], mags_.pressGamma[1], "%.2f"))
	{
		const bool changed = !Base::ApproxEqu(temp_.pressGamma, simParams.pressGamma);

		if (changed && liveParamEditing_) // apply param changes immediately
		{
			simParamsMgr_.Set_PressGamma(temp_.pressGamma);
		}

		// mark unsaved changes when not live editing
		changed_.pressGamma = changed && !liveParamEditing_;
		unsavedParamChanges_ = changed && !liveParamEditing_;
	}
	if (cached) { ImGui::PopStyleColor(); }
}

void UserInterfaceMgr::BuildParamElement_JacobiOmega(const Sim::SimParams& simParams)
{
	const bool cached = changed_.jacobiOmega;
	if (cached) { ImGui::PushStyleColor(ImGuiCol_Text, RGBA_unsaved_); }
	ImGui::SetNextItemWidth(170);
	if (ImGui::InputFloat(" jacobi omega##input_jacobiOmega",
		&temp_.jacobiOmega, mags_.jacobiOmega[0], mags_.jacobiOmega[1], "%.2f"))
	{
		const bool changed = !Base::ApproxEqu(temp_.jacobiOmega, simParams.jacobiOmega);

		if (changed && liveParamEditing_) // apply param changes immediately
		{
			simParamsMgr_.Set_JacobiOmega(temp_.jacobiOmega);
		}

		// mark unsaved changes when not live editing
		changed_.jacobiOmega = changed && !liveParamEditing_;
		unsavedParamChanges_ = changed && !liveParamEditing_;
	}
	if (cached) { ImGui::PopStyleColor(); }
}

void UserInterfaceMgr::BuildParamElement_DensityDevTarget(const Sim::SimParams& simParams)
{
	const bool cached = changed_.densityDevTarget;
	if (cached) { ImGui::PushStyleColor(ImGuiCol_Text, RGBA_unsaved_); }
	ImGui::SetNextItemWidth(170);
	if (ImGui::InputFloat(" density dev target##input_densityDevTarget",
		&temp_.densityDevTarget, mags_.densityDevTarget[0], mags_.densityDevTarget[1], "%.3f"))
	{
		const bool changed = !Base::ApproxEqu(temp_.densityDevTarget, simParams.densityDevTarget);
		if (changed && liveParamEditing_) // apply param changes immediately
		{
			simParamsMgr_.Set_DensityDevTarget(temp_.densityDevTarget);
		}
		// mark unsaved changes when not live editing
		changed_.densityDevTarget = changed && !liveParamEditing_;
		unsavedParamChanges_ = changed && !liveParamEditing_;
	}
	if (cached) { ImGui::PopStyleColor(); }
}

void UserInterfaceMgr::BuildParamElement_MinSolverIters(const Sim::SimParams& simParams)
{
	const bool cached = changed_.minSolverIters;
	if (cached) { ImGui::PushStyleColor(ImGuiCol_Text, RGBA_unsaved_); }
	ImGui::SetNextItemWidth(170);
	if (ImGui::InputInt(" min solver iters##input_minSolverIters",
		&temp_.minSolverIters, mags_.minSolverIters[0], mags_.minSolverIters[1]))
	{
		// clamp to non-zero, non-negative
		temp_.minSolverIters = std::max(temp_.minSolverIters, 1);

		const bool changed = (static_cast<uint32_t>(temp_.minSolverIters) != simParams.minSolverIters);
		if (changed && liveParamEditing_) // apply param changes immediately
		{
			simParamsMgr_.Set_MinSolverIters(static_cast<uint32_t>(temp_.minSolverIters));
		}
		// mark unsaved changes when not live editing
		changed_.minSolverIters = changed && !liveParamEditing_;
		unsavedParamChanges_ = changed && !liveParamEditing_;
	}
	if (cached) { ImGui::PopStyleColor(); }
}

void UserInterfaceMgr::BuildParamElement_MaxSolverIters(const Sim::SimParams& simParams)
{
	const bool cached = changed_.maxSolverIters;
	if (cached) { ImGui::PushStyleColor(ImGuiCol_Text, RGBA_unsaved_); }
	ImGui::SetNextItemWidth(170);
	if (ImGui::InputInt(" max solver iters##input_maxSolverIters",
		&temp_.maxSolverIters, mags_.maxSolverIters[0], mags_.maxSolverIters[1]))
	{
		// clamp to non-zero, non-negative
		temp_.maxSolverIters = std::max(temp_.maxSolverIters, 1);

		const bool changed = (static_cast<uint32_t>(temp_.maxSolverIters) != simParams.maxSolverIters);
		if (changed && liveParamEditing_) // apply param changes immediately
		{
			simParamsMgr_.Set_MaxSolverIters(static_cast<uint32_t>(temp_.maxSolverIters));
		}
		// mark unsaved changes when not live editing
		changed_.maxSolverIters = changed && !liveParamEditing_;
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

// --- [ helpers ] --------------------------------------------------------------------------------

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

	if (changed_.timeStep)
	{
		simParamsMgr_.Set_TimeStep(temp_.timeStep);
		changed_.timeStep = false;
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
	if (changed_.pressGamma)
	{
		simParamsMgr_.Set_PressGamma(temp_.pressGamma);
		changed_.pressGamma = false;
	}
	if (changed_.jacobiOmega)
	{
		simParamsMgr_.Set_JacobiOmega(temp_.jacobiOmega);
		changed_.jacobiOmega = false;
	}
	if (changed_.densityDevTarget)
	{
		simParamsMgr_.Set_DensityDevTarget(temp_.densityDevTarget);
		changed_.densityDevTarget = false;
	}
	if (changed_.minSolverIters)
	{
		simParamsMgr_.Set_MinSolverIters(static_cast<uint32_t>(temp_.minSolverIters));
		changed_.minSolverIters = false;
	}
	if (changed_.maxSolverIters)
	{
		simParamsMgr_.Set_MaxSolverIters(static_cast<uint32_t>(temp_.maxSolverIters));
		changed_.maxSolverIters = false;
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

	if (changed_.timeStep)
	{
		temp_.timeStep = simParams.timeStep;
		changed_.timeStep = false;
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
	if (changed_.pressGamma)
	{
		temp_.pressGamma = simParams.pressGamma;
		changed_.pressGamma = false;
	}
	if (changed_.jacobiOmega)
	{
		temp_.jacobiOmega = simParams.jacobiOmega;
		changed_.jacobiOmega = false;
	}
	if (changed_.densityDevTarget)
	{
		temp_.densityDevTarget = simParams.densityDevTarget;
		changed_.densityDevTarget = false;
	}
	if (changed_.minSolverIters)
	{
		temp_.minSolverIters = static_cast<int>(simParams.minSolverIters);
		changed_.minSolverIters = false;
	}
	if (changed_.maxSolverIters)
	{
		temp_.maxSolverIters = static_cast<int>(simParams.maxSolverIters);
		changed_.maxSolverIters = false;
	}
	if (changed_.gravity)
	{
		temp_.gravity = simParams.gravity.y;
		changed_.gravity = false;
	}

	unsavedParamChanges_ = false;
}

inline float UserInterfaceMgr::ReadStatData(
	const Sync::StatSnapshot* snap, Stats::StatMetric metric) const
{
	if (!snap)
	{
		SPDLOG_ERROR("Failed to read stat data (no snap provided)");
	}
	const uint16_t i = Stats::GetIdx(metric);

	switch (statTableView_)
	{
	case StatTableView::Latest: { return snap->latest[i]; }
	case StatTableView::Avg: { return snap->avg[i]; }
	case StatTableView::Min: { return snap->min[i]; }
	case StatTableView::Max: { return snap->max[i]; }
	}

	SPDLOG_WARN("Invalid StatTableView enum value");
	return 0;
}

inline uint64_t UserInterfaceMgr::ReadOpTiming(
	const Sync::BenchSnapshot* snap, Bench::BenchOp op) const
{
	if (!snap)
	{
		SPDLOG_ERROR("Failed to read operation timing (no snap provided)");
	}
	const uint16_t i = Bench::GetIdx(op);

	switch (benchTableView_)
	{
	case BenchTableView::Latest: { return snap->latest[i]; }
	case BenchTableView::Avg_All: { return snap->avgAll[i]; }
	case BenchTableView::Avg_Used: { return snap->avgUsed[i]; }
	case BenchTableView::Min: { return snap->min[i]; }
	case BenchTableView::Max: { return snap->max[i]; }
	}

	SPDLOG_WARN("Invalid BenchTableView enum value");
	return 0;
}

inline uint64_t UserInterfaceMgr::ReadOpGroup_Min(
	const Sync::BenchSnapshot* snap, const Bench::BenchOpGroupDef& opGroup) const
{
	uint64_t res = std::numeric_limits<uint64_t>::max();

	// pick one of the operation timings as a representative
	// (they should all be the same value, but the groups are variably-sized)
	for (const auto& op : opGroup.ops)
	{
		res = std::min(res, ReadOpTiming(snap, op));
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
	const float x = ImGui::GetCursorPosX() + fmaxf(0.0f, avail - textWidth - 2.0f);

	ImGui::SetCursorPosX(x);
	ImGui::TextUnformatted(buf);
}

// formats a stat value according to its metric definition
inline void UserInterfaceMgr::FormatStatValue(
	const Stats::StatMetricDef& def,
	char* out,
	float x,
	uint64_t cap)
{
	// ints
	if (def.var == "int")
	{
		int64_t ix = static_cast<int64_t>(std::llround(x));

		if (def.unit.empty())
		{
			snprintf(out, cap, "%lld", static_cast<long long>(ix));
		}
		else
		{
			snprintf(out, cap, "%lld %.*s",
				static_cast<long long>(ix),
				static_cast<int>(def.unit.size()), def.unit.data());
		}
		return;
	}

	// floats
	if (def.unit == "%")
	{
		snprintf(out, cap, "%.2f %%", x * 100.0f);
	}
	else if (def.unit == "m/s" && false) // TODO: use meters per second?
	{
		snprintf(out, cap, "%.2f m/s", x);
	}
	else
	{
		snprintf(out, cap, "%.2f", x);
	}
}

// --- [ dockspace ] ------------------------------------------------------------------------------

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
