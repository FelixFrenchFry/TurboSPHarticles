// parameters_vis.cpp

// external
#include <spdlog/spdlog.h>

// internal
#include "parameters_vis.h"



using namespace Vis;

VisParamsMgr::VisParamsMgr()
{
	// TODO
}

std::pair<VisParams, uint64_t> VisParamsMgr::SnapshotParams() const
{
	while (true)
	{
		// check for ongoing write
		uint64_t before = version_.load();
		if (before % 2 != 0) { continue; }

		// copy parameters
		VisParams res = params_;

		// make sure no write happened during copy
		uint64_t after = version_.load();
		if (before == after) { return { res, after }; }
	}
}

uint64_t VisParamsMgr::StableVersion() const
{
	while (true)
	{
		// make sure no write is ongoing
		uint64_t a = version_.load();
		if (a % 2 == 0)
		{
			uint64_t b = version_.load();
			if (a == b) { return a; }
		}
	}
}

void VisParamsMgr::SetParam(WindowPatch patch)
{
	// apply changes
	BeginWrite();
	if (patch.size) { params_.window.size = *patch.size; }
	if (patch.title) { params_.window.title = *patch.title; }
	if (patch.resizable) { params_.window.resizable = *patch.resizable; }
	if (patch.vsync) { params_.window.vsync = *patch.vsync; }
	EndWrite();
}

void VisParamsMgr::SetParam(FrameBufferPatch patch)
{
	// apply changes
	BeginWrite();
	if (patch.size) { params_.frameBuffer.size = *patch.size; }
	if (patch.dpiScale) { params_.frameBuffer.dpiScale = *patch.dpiScale; }
	EndWrite();
}

void VisParamsMgr::SetParam(CameraPatch patch)
{
	// apply changes
	BeginWrite();
	if (patch.viewMat) { params_.camera.viewMat = *patch.viewMat; }
	if (patch.projMat) { params_.camera.projMat = *patch.projMat; }
	if (patch.camPos) { params_.camera.camPos = *patch.camPos; }
	EndWrite();
}

void VisParamsMgr::SetParam(RenderPatch patch)
{
	// apply changes
	BeginWrite();
	if (patch.lightDirVS) { params_.render.lightDirVS = *patch.lightDirVS; }
	if (patch.baseColor) { params_.render.baseColor = *patch.baseColor; }
	if (patch.alpha) { params_.render.alpha = *patch.alpha; }
	if (patch.radius) { params_.render.radius = *patch.radius; }
	if (patch.scalarMin) { params_.render.scalarMin = *patch.scalarMin; }
	if (patch.scalarMax) { params_.render.scalarMax = *patch.scalarMax; }
	if (patch.useScalar) { params_.render.useScalar = *patch.useScalar; }
	EndWrite();
}

void VisParamsMgr::BeginWrite() { version_.fetch_add(1); }

void VisParamsMgr::EndWrite() { version_.fetch_add(1); }
