// parameters_vis.cpp

// external
#include <spdlog/spdlog.h>
#include <thread>

// internal
#include "Source/Base/helpers.h"
#include "parameters_vis.h"



using namespace Vis;

std::pair<VisParams, int64_t> VisParamsMgr::SnapshotParams() const
{
	while (true)
	{
		// check for ongoing write
		int64_t before = version_.load(std::memory_order_acquire);
		if (before % 2 != 0) { std::this_thread::yield(); continue; }

		// copy parameters
		VisParams res = params_;

		// make sure no write happened during copy
		int64_t after = version_.load(std::memory_order_acquire);
		if (before == after) { return { res, after }; }

		// else try again
		std::this_thread::yield();
	}
}

int64_t VisParamsMgr::StableVersion() const
{
	while (true)
	{
		// make sure no write is ongoing
		int64_t a = version_.load(std::memory_order_acquire);
		if (a % 2 == 0)
		{
			int64_t b = version_.load(std::memory_order_acquire);
			if (a == b) { return a; }
		}

		// else try again
		std::this_thread::yield();
	}
}

void VisParamsMgr::SetParam(const WindowPatch& patch)
{
	// detect if any parameter would actually change
	bool changed = false;

	Base::MarkIfChanged(patch.size, params_.window.size, changed);
	Base::MarkIfChanged(patch.title, params_.window.title, changed);
	Base::MarkIfChanged(patch.resizable, params_.window.resizable, changed);
	Base::MarkIfChanged(patch.vsync, params_.window.vsync, changed);

	if (!changed) { return; } // nothing to change

	// apply changes
	BeginWrite();
	if (patch.size) { params_.window.size = *patch.size; }
	if (patch.title) { params_.window.title = *patch.title; }
	if (patch.resizable) { params_.window.resizable = *patch.resizable; }
	if (patch.vsync) { params_.window.vsync = *patch.vsync; }
	EndWrite();
}

void VisParamsMgr::SetParam(const FrameBufferPatch& patch)
{
	// detect if any parameter would actually change
	bool changed = false;

	Base::MarkIfChanged(patch.size, params_.frameBuffer.size, changed);
	Base::MarkIfChanged(patch.dpiScale, params_.frameBuffer.dpiScale, changed);

	if (!changed) { return; } // nothing to change

	// apply changes
	BeginWrite();
	if (patch.size) { params_.frameBuffer.size = *patch.size; }
	if (patch.dpiScale) { params_.frameBuffer.dpiScale = *patch.dpiScale; }
	EndWrite();
}

void VisParamsMgr::SetParam(const CameraPatch& patch)
{
	// detect if any parameter would actually change
	bool changed = false;

	Base::MarkIfChanged(patch.viewMat, params_.camera.viewMat, changed);
	Base::MarkIfChanged(patch.projMat, params_.camera.projMat, changed);
	Base::MarkIfChanged(patch.camPos, params_.camera.camPos, changed);
	Base::MarkIfChanged(patch.yawDeg, params_.camera.yawDeg, changed);
	Base::MarkIfChanged(patch.pitchDeg, params_.camera.pitchDeg, changed);
	Base::MarkIfChanged(patch.moveSpeed, params_.camera.moveSpeed, changed);
	Base::MarkIfChanged(patch.turnSpeedDeg, params_.camera.turnSpeedDeg, changed);
	Base::MarkIfChanged(patch.yFovDeg, params_.camera.yFovDeg, changed);
	Base::MarkIfChanged(patch.zNear, params_.camera.zNear, changed);
	Base::MarkIfChanged(patch.zFar, params_.camera.zFar, changed);

	if (!changed) { return; } // nothing to change

	// apply changes
	BeginWrite();
	if (patch.viewMat) { params_.camera.viewMat = *patch.viewMat; }
	if (patch.projMat) { params_.camera.projMat = *patch.projMat; }
	if (patch.camPos) { params_.camera.camPos = *patch.camPos; }
	if (patch.yawDeg) { params_.camera.yawDeg = *patch.yawDeg; }
	if (patch.pitchDeg) { params_.camera.pitchDeg = *patch.pitchDeg; }
	if (patch.moveSpeed) { params_.camera.moveSpeed = *patch.moveSpeed; }
	if (patch.turnSpeedDeg) { params_.camera.turnSpeedDeg = *patch.turnSpeedDeg; }
	if (patch.yFovDeg) { params_.camera.yFovDeg = *patch.yFovDeg; }
	if (patch.zNear) { params_.camera.zNear = *patch.zNear; }
	if (patch.zFar) { params_.camera.zFar = *patch.zFar; }
	EndWrite();
}

void VisParamsMgr::SetParam(const RenderPatch& patch)
{
	// detect if any parameter would actually change
	bool changed = false;

	Base::MarkIfChanged(patch.lightDirVS, params_.render.lightDirVS, changed);
	Base::MarkIfChanged(patch.baseColor, params_.render.baseColor, changed);
	Base::MarkIfChanged(patch.rigidBoundaryColor, params_.render.rigidBoundaryColor, changed);
	Base::MarkIfChanged(patch.centerOfMassXAxisColor, params_.render.centerOfMassXAxisColor, changed);
	Base::MarkIfChanged(patch.centerOfMassYAxisColor, params_.render.centerOfMassYAxisColor, changed);
	Base::MarkIfChanged(patch.centerOfMassZAxisColor, params_.render.centerOfMassZAxisColor, changed);
	Base::MarkIfChanged(patch.alpha, params_.render.alpha, changed);
	Base::MarkIfChanged(patch.centerOfMassXRayAlpha, params_.render.centerOfMassXRayAlpha, changed);
	Base::MarkIfChanged(patch.scalarMin, params_.render.scalarMin, changed);
	Base::MarkIfChanged(patch.scalarMax, params_.render.scalarMax, changed);
	Base::MarkIfChanged(patch.useScalar, params_.render.useScalar, changed);
	Base::MarkIfChanged(patch.showFluidParticles, params_.render.showFluidParticles, changed);
	Base::MarkIfChanged(patch.showRigidBoundaryParticles, params_.render.showRigidBoundaryParticles, changed);
	Base::MarkIfChanged(patch.showRigidBodyCentersOfMass, params_.render.showRigidBodyCentersOfMass, changed);
	Base::MarkIfChanged(patch.rigidBoundaryRadiusScale, params_.render.rigidBoundaryRadiusScale, changed);

	if (!changed) { return; } // nothing to change

	// apply changes
	BeginWrite();
	if (patch.lightDirVS) { params_.render.lightDirVS = *patch.lightDirVS; }
	if (patch.baseColor) { params_.render.baseColor = *patch.baseColor; }
	if (patch.rigidBoundaryColor) { params_.render.rigidBoundaryColor = *patch.rigidBoundaryColor; }
	if (patch.centerOfMassXAxisColor) { params_.render.centerOfMassXAxisColor = *patch.centerOfMassXAxisColor; }
	if (patch.centerOfMassYAxisColor) { params_.render.centerOfMassYAxisColor = *patch.centerOfMassYAxisColor; }
	if (patch.centerOfMassZAxisColor) { params_.render.centerOfMassZAxisColor = *patch.centerOfMassZAxisColor; }
	if (patch.alpha) { params_.render.alpha = *patch.alpha; }
	if (patch.centerOfMassXRayAlpha) { params_.render.centerOfMassXRayAlpha = *patch.centerOfMassXRayAlpha; }
	if (patch.scalarMin) { params_.render.scalarMin = *patch.scalarMin; }
	if (patch.scalarMax) { params_.render.scalarMax = *patch.scalarMax; }
	if (patch.useScalar) { params_.render.useScalar = *patch.useScalar; }
	if (patch.showFluidParticles) { params_.render.showFluidParticles = *patch.showFluidParticles; }
	if (patch.showRigidBoundaryParticles) { params_.render.showRigidBoundaryParticles = *patch.showRigidBoundaryParticles; }
	if (patch.showRigidBodyCentersOfMass) { params_.render.showRigidBodyCentersOfMass = *patch.showRigidBodyCentersOfMass; }
	if (patch.rigidBoundaryRadiusScale) { params_.render.rigidBoundaryRadiusScale = *patch.rigidBoundaryRadiusScale; }
	EndWrite();
}

void VisParamsMgr::BeginWrite() { version_.fetch_add(1); }

void VisParamsMgr::EndWrite() { version_.fetch_add(1); }
