// user_interface.cpp

// external
#include <algorithm>
#include <cmath>
#include <cinttypes>
#include <cstdarg>
#include <cstdio>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <spdlog/spdlog.h>
#include <stdexcept>

// internal
#include "source/base/helpers.h"
#include "source/visualization/fonts/iosevka_compressed.h"
#include "source/visualization/render/rendering.h"
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
		throw std::runtime_error(
			"Cannot initialize user interface without a window and GLSL version.");
	}

	window_ = window;
	iniFilename_ = iniFilename ? iniFilename : "";

	// create ImGui context
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
		.gravity = simParams.gravity.y,
		.pressGamma = simParams.pressGamma,
		.jacobiOmega = simParams.jacobiOmega,
		.densityDevTarget = simParams.densityDevTarget,
		.minSolverIters = static_cast<int>(simParams.minSolverIters),
		.maxSolverIters = static_cast<int>(simParams.maxSolverIters)
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

	// ImGui second
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
		BuildSection_SceneManagement(simulationMgr, visParams);
	}

	if (ImGui::CollapsingHeader("Simulation Control", ImGuiTreeNodeFlags_DefaultOpen))
	{
		BuildSection_SimulationControl(simulationMgr, simParams);
	}

	if (ImGui::CollapsingHeader("Simulation Parameters"))
	{
		BuildSection_SimulationParameters(simParams);
	}

	if (ImGui::CollapsingHeader("Visualization Settings"))
	{
		BuildSection_VisualizationSettings(visParams);
	}

	if (ImGui::CollapsingHeader("Simulation Statistics"))
	{
		BuildSection_SimulationStatistics(statSnap);
	}

	if (ImGui::CollapsingHeader("Performance Benchmarks"))
	{
		Bench::BenchOpList benchOps;
		simulationMgr.GetBenchOps(benchOps);
		BuildSection_PerformanceBenchmarks(benchSnap, benchOps);
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
		simulationMgr.ResetAndRebuildSimState();

		// TODO: reset statistics and benchmarks
	}
	if (cached) { ImGui::EndDisabled(); cached = false; }
	ImGui::PopStyleColor(1);

	if (ImGui::TreeNode("scene settings"))
	{
		ImGui::Spacing();

		bool showParticlesF = visParams.render.showParticlesF;
		if (ImGui::Checkbox("show fluid particles",
			&showParticlesF))
		{
			visParamsMgr_.SetParam(Vis::RenderPatch
				{
					.showParticlesF = showParticlesF
				});
		}

		bool showParticlesR = visParams.render.showParticlesR;
		if (ImGui::Checkbox("show rigid particles",
			&showParticlesR))
		{
			visParamsMgr_.SetParam(Vis::RenderPatch
				{
					.showParticlesR = showParticlesR
				});
		}

		bool showRigidBodyCentersOfMass = visParams.render.showRigidBodyCentersOfMass;
		if (ImGui::Checkbox("show rigid body centers of mass",
			&showRigidBodyCentersOfMass))
		{
			visParamsMgr_.SetParam(Vis::RenderPatch
				{
					.showRigidBodyCentersOfMass = showRigidBodyCentersOfMass
				});
		}

		ImGui::TreePop();
	}
	ImGui::Spacing();
}

void UserInterfaceMgr::BuildSection_VisualizationSettings(
	const Vis::VisParams& visParams)
{
	ImGui::Spacing();

	if (ImGui::TreeNode("color coding"))
	{
		// select fluid particle color coding mode
		bool none = visParams.render.scalarModeF == Base::ColorCodingMode::None;
		if (ImGui::Checkbox("##checkbox_scalarModeNone", &none) && none)
		{
			visParamsMgr_.SetParam(Vis::RenderPatch
				{
					.scalarModeF = Base::ColorCodingMode::None
				});
		}
		ImGui::SameLine();

		// adjust fluid particle opacity
		float alpha = visParams.render.alpha;
		ImGui::SetNextItemWidth(170.0f);
		if (ImGui::InputFloat(" none##input_alpha",
			&alpha, 0.1f, 0.1f, "%.2f"))
		{
			visParamsMgr_.SetParam(Vis::RenderPatch
				{
					.alpha = std::clamp(alpha, 0.0f, 1.0f)
				});
		}

		bool velocityMag =
			visParams.render.scalarModeF == Base::ColorCodingMode::VelocityMag;
		if (ImGui::Checkbox("##checkbox_scalarModeVelocityMag", &velocityMag))
		{
			visParamsMgr_.SetParam(Vis::RenderPatch
				{
					.scalarModeF = velocityMag
						? Base::ColorCodingMode::VelocityMag
						: Base::ColorCodingMode::None
				});
		}
		ImGui::SameLine();

		// adjust upper velocity magnitude mapped to the color map
		float scalarMax = visParams.render.scalarMax;
		ImGui::SetNextItemWidth(170.0f);
		if (ImGui::InputFloat(" velocity mag##input_scalarMax",
			&scalarMax, 0.1f, 1.0f, "%.2f"))
		{
			visParamsMgr_.SetParam(Vis::RenderPatch
				{
					.scalarMax = std::max(scalarMax, 0.01f)
				});
		}

		ImGui::TreePop();
	}

	if (ImGui::TreeNode("color gradient"))
	{
		const auto& colorMapTurbo = Render::RenderingMgr::GetColorMap_Turbo();
		const auto& colorMapViridis = Render::RenderingMgr::GetColorMap_Viridis();
		const auto& colorMapInferno = Render::RenderingMgr::GetColorMap_Inferno();
		const auto& colorMapFast = Render::RenderingMgr::GetColorMap_Fast();
		const auto& colorMapCoolwarm = Render::RenderingMgr::GetColorMap_Coolwarm();
		const auto& colorMapBlues = Render::RenderingMgr::GetColorMap_Blues();
		const auto& colorMapAurora = Render::RenderingMgr::GetColorMap_Aurora();
		const auto& colorMapEarth = Render::RenderingMgr::GetColorMap_Earth();

		// select color map and show its gradient
		BuildColorMapElement("turbo", Base::ColorMap::Turbo,
			colorMapTurbo, visParams);
		BuildColorMapElement("viridis", Base::ColorMap::Viridis,
			colorMapViridis, visParams);
		BuildColorMapElement("inferno", Base::ColorMap::Inferno,
			colorMapInferno, visParams);
		BuildColorMapElement("fast", Base::ColorMap::Fast,
			colorMapFast, visParams);
		BuildColorMapElement("coolwarm", Base::ColorMap::Coolwarm,
			colorMapCoolwarm, visParams);
		BuildColorMapElement("blues", Base::ColorMap::Blues,
			colorMapBlues, visParams);
		BuildColorMapElement("aurora", Base::ColorMap::Aurora,
			colorMapAurora, visParams);
		BuildColorMapElement("earth", Base::ColorMap::Earth,
			colorMapEarth, visParams);

		ImGui::TreePop();
	}
	ImGui::Spacing();
}

void UserInterfaceMgr::BuildColorMapElement(
	const char* label,
	Base::ColorMap colorMap,
	const Render::ColorMapData& colorMapValues,
	const Vis::VisParams& visParams)
{
	ImGui::PushID(label);

	bool selected = visParams.render.colorMap == colorMap;
	if (ImGui::Checkbox("##checkbox", &selected) && selected)
	{
		visParamsMgr_.SetParam(Vis::RenderPatch
			{
				.colorMap = colorMap
			});
	}
	ImGui::SameLine();

	// draw color map preview from its lookup-table entries
	const ImVec2 previewMin = ImGui::GetCursorScreenPos();
	const ImVec2 previewSize{ 225.0f, ImGui::GetFrameHeight() };
	ImGui::Dummy(previewSize);
	ImDrawList* drawList = ImGui::GetWindowDrawList();
	for (uint32_t i = 0; i < colorMapValues.size() - 1; i++)
	{
		const float x0 = previewMin.x + previewSize.x *
			static_cast<float>(i) / static_cast<float>(colorMapValues.size() - 1);
		const float x1 = previewMin.x + previewSize.x *
			static_cast<float>(i + 1) / static_cast<float>(colorMapValues.size() - 1);
		const auto& color0 = colorMapValues[i];
		const auto& color1 = colorMapValues[i + 1];
		const ImU32 colorLeft = IM_COL32(color0.r, color0.g, color0.b, 255);
		const ImU32 colorRight = IM_COL32(color1.r, color1.g, color1.b, 255);
		drawList->AddRectFilledMultiColor(
			ImVec2{ x0, previewMin.y },
			ImVec2{ x1, previewMin.y + previewSize.y },
			colorLeft, colorRight, colorRight, colorLeft);
	}

	ImGui::SameLine();
	ImGui::Text(" %s", label);
	ImGui::PopID();
}

void UserInterfaceMgr::BuildSection_SimulationControl(
	const Sim::SimulationMgr& simulationMgr,
	const Sim::SimParams& simParams)
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
	snprintf(stringBuffer, sizeof(stringBuffer), "%" PRIu64, simulationMgr.simStep_);
	ImGui::Text(stringBuffer);
	ImGui::Spacing();
}

void UserInterfaceMgr::BuildSection_SimulationParameters(
	const Sim::SimParams& simParams)
{
	ImGui::Spacing();

	BuildParamElement_TimeStep(simParams);
	BuildParamElement_ViscosityF(simParams);
	BuildParamElement_ViscosityB(simParams);
	BuildParamElement_Gravity(simParams);
	BuildParamElement_PressGamma(simParams);
	BuildParamElement_JacobiOmega(simParams);
	BuildParamElement_DensityDevTarget(simParams);
	BuildParamElement_MinSolverIters(simParams);
	BuildParamElement_MaxSolverIters(simParams);

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
	const Sync::StatSnapshot* statSnap)
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
				const double val = ReadStatData(statSnap, metric);

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
			Stats::StatMetric::NumR,
			Stats::StatMetric::NumNbrFF_Avg,
			Stats::StatMetric::NumNbrFF_Min,
			Stats::StatMetric::NumNbrFF_Max,
			Stats::StatMetric::NumNbrFF_P99,
			Stats::StatMetric::NumNbrFB_Avg,
			Stats::StatMetric::NumNbrFB_Min,
			Stats::StatMetric::NumNbrFB_Max,
			Stats::StatMetric::NumNbrFB_P99,
			Stats::StatMetric::NumNbrFR_Avg,
			Stats::StatMetric::NumNbrFR_Min,
			Stats::StatMetric::NumNbrFR_Max,
			Stats::StatMetric::NumNbrFR_P99,
			Stats::StatMetric::NumNbrRF_Avg,
			Stats::StatMetric::NumNbrRF_Min,
			Stats::StatMetric::NumNbrRF_Max,
			Stats::StatMetric::NumNbrRF_P99,
			Stats::StatMetric::NumNbrRB_Avg,
			Stats::StatMetric::NumNbrRB_Min,
			Stats::StatMetric::NumNbrRB_Max,
			Stats::StatMetric::NumNbrRB_P99,
			Stats::StatMetric::NumNbrRR_Avg,
			Stats::StatMetric::NumNbrRR_Min,
			Stats::StatMetric::NumNbrRR_Max,
			Stats::StatMetric::NumNbrRR_P99
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
	const Sync::BenchSnapshot* benchSnap,
	const Bench::BenchOpList& benchOps)
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

	// percentages are meaningful only when all rows use the same step set
	const bool showPercent =
		benchTableView_ == BTV::Latest ||
		benchTableView_ == BTV::Avg_All;

	// --- [ per-operation timing and totals ] -------------------------------------------------------

	// timing per operation (row)
	std::vector<uint64_t> opTimings(benchOps.size(), 0);

	// sum of displayed non-total rows, used to derive unaccounted time
	uint64_t accountedTime = 0;
	int totalRowIdx = -1;

	for (uint32_t i = 0; i < benchOps.size(); i++)
	{
		const Bench::BenchOp op = benchOps[i];

		const uint64_t t = ReadOpTiming(benchSnap, op);
		opTimings[i] = t;

		//SPDLOG_ERROR("time: {}", t);

		// detect BenchOp::Total row to exclude from percentage calculations
		if (op == Bench::BenchOp::Total)
		{
			totalRowIdx = static_cast<int>(i);
		}
		else
		{
			accountedTime += t;
		}
	}

	const uint64_t totalTime = totalRowIdx >= 0
		? opTimings[static_cast<std::size_t>(totalRowIdx)]
		: accountedTime;
	const bool showOther =
		totalRowIdx >= 0 &&
		(benchTableView_ == BTV::Latest ||
		 benchTableView_ == BTV::Avg_All);

	auto SaturatingSubtract = [](const uint64_t a, const uint64_t b)
	{
		return (a > b) ? (a - b) : 0ull;
	};

	// --- [ benchmark table ] -----------------------------------------------------------------------

	const int numCols = showPercent ? 3 : 2;

	if (ImGui::BeginTable("table_bench", numCols,
		ImGuiTableFlags_RowBg |
		ImGuiTableFlags_BordersH |
		ImGuiTableFlags_Resizable |
		ImGuiTableFlags_SizingStretchProp))
	{
		// column headers
		ImGui::TableSetupColumn("operation",
			ImGuiTableColumnFlags_WidthStretch, 0.65f);
		ImGui::TableSetupColumn(Bench::UnitLabel(benchTimeUnit_),
			ImGuiTableColumnFlags_WidthStretch, 0.20f);
		if (showPercent)
		{
			ImGui::TableSetupColumn("%",
				ImGuiTableColumnFlags_WidthStretch, 0.15f);
		}
		ImGui::TableHeadersRow();

		auto DrawPercentBar = [&](const uint64_t t, const bool isTotal) -> void
		{
			if (!isTotal && totalTime > 0)
			{
				const float pct = 100.0f * (static_cast<float>(t)
								/ static_cast<float>(totalTime));
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
		};

		auto DrawPlainTimingRow = [&](const char* label, const uint64_t t, const bool isTotal) -> void
		{
			ImGui::TableNextRow();

			if (isTotal)
			{
				ImGui::TableSetBgColor(ImGuiTableBgTarget_RowBg0, IM_COL32(200, 50, 50, 255));
				ImGui::TableSetBgColor(ImGuiTableBgTarget_RowBg1, IM_COL32(200, 50, 50, 255));
			}

			ImGui::TableSetColumnIndex(0);
			ImGui::TextUnformatted(label);

			ImGui::TableSetColumnIndex(1);
			TableTextRightAligned("%.2f", Bench::ConvertFromNs(benchTimeUnit_, t));

			if (showPercent)
			{
				ImGui::TableSetColumnIndex(2);
				DrawPercentBar(t, isTotal);
			}
		};

		auto DrawOpRow = [&](uint32_t rowIdx) -> void
		{
			const Bench::BenchOp op = benchOps[rowIdx];
			const auto& def = Bench::Definitions[Bench::GetIdx(op)];
			const uint64_t t = opTimings[rowIdx];
			const double t_unit = Bench::ConvertFromNs(benchTimeUnit_, t);

			ImGui::TableNextRow();

			// column 0 with alias
			ImGui::TableSetColumnIndex(0);
			ImGui::TextUnformatted(def.alias.data());
			if (ImGui::IsItemHovered())
			{
				ImGui::SetTooltip("%s", def.name.data());
			}

			// column 1 with timing in selected unit
			ImGui::TableSetColumnIndex(1);
			TableTextRightAligned("%.2f", t_unit);

			// column 2 with percentages of total sim time
			if (showPercent)
			{
				ImGui::TableSetColumnIndex(2);
				DrawPercentBar(t, false);
			}
		};

		// draw all regular rows first
		for (uint32_t i = 0; i < benchOps.size(); i++)
		{
			if (static_cast<int>(i) == totalRowIdx) { continue; }

			DrawOpRow(i);
		}

		// draw BenchOp::Total row last (if present)
		if (totalRowIdx >= 0)
		{
			if (showOther)
			{
				DrawPlainTimingRow("other",
					SaturatingSubtract(totalTime, accountedTime), false);
			}

			DrawPlainTimingRow("total", totalTime, true);
		}

		ImGui::EndTable();
	}

	// TODO: sparkline?
	// sparkline for BenchOp::Total

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

// --- [ helpers ] --------------------------------------------------------------------------------

void UserInterfaceMgr::CreateContexts(const char* iniFilename)
{
	// ImGui context
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();

	ImGuiIO& io = ImGui::GetIO();
	io.IniFilename = iniFilename && *iniFilename ? iniFilename : nullptr;
	io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
	io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
}

void UserInterfaceMgr::DestroyContexts()
{
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
		SPDLOG_ERROR("Failed to load custom compressed font from header file.");
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
	if (changed_.gravity)
	{
		simParamsMgr_.Set_Gravity(glm::vec3{ 0.0f, temp_.gravity, 0.0f });
		changed_.gravity = false;
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
	if (changed_.gravity)
	{
		temp_.gravity = simParams.gravity.y;
		changed_.gravity = false;
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

	unsavedParamChanges_ = false;
}

inline double UserInterfaceMgr::ReadStatData(
	const Sync::StatSnapshot* snap, Stats::StatMetric metric) const
{
	if (!snap)
	{
		SPDLOG_ERROR("Cannot read statistics data without a snapshot.");
		return 0.0f;
	}
	const uint16_t i = Stats::GetIdx(metric);

	switch (statTableView_)
	{
	case StatTableView::Latest: { return snap->latest[i]; }
	case StatTableView::Avg: { return snap->avg[i]; }
	case StatTableView::Min: { return snap->min[i]; }
	case StatTableView::Max: { return snap->max[i]; }
	}

	SPDLOG_WARN("Invalid StatTableView value.");
	return 0;
}

inline uint64_t UserInterfaceMgr::ReadOpTiming(
	const Sync::BenchSnapshot* snap, Bench::BenchOp op) const
{
	if (!snap)
	{
		SPDLOG_ERROR("Cannot read operation timing without a snapshot.");
		return 0;
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

	SPDLOG_WARN("Invalid BenchTableView value.");
	return 0;
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
	double x,
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
