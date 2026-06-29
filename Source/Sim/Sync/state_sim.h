// state_sim.h
//
// TODO: comment

#pragma once

// external
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>



namespace Sync
{
	inline constexpr uint32_t InvalidRigidBodyId = std::numeric_limits<uint32_t>::max();
	inline constexpr uint32_t InvalidRigidSampleIdx = std::numeric_limits<uint32_t>::max();

	// view into fluid particle data
	// (non-owning, for read-only access)
	struct FluidView
	{
		// metadata
		uint32_t N_ = 0;

		// fields
		std::span<uint32_t> id;
		std::span<float> xPos, yPos, zPos;
		std::span<float> xVel, yVel, zVel;
		std::span<float> xVelPred, yVelPred, zVelPred;
		std::span<float> xAccPress, yAccPress, zAccPress;
		std::span<float> sourceTerm;
		std::span<float> diagElem;
		std::span<float> pressure;
		std::span<float> density;
	};

	// view into boundary particle data
	// (non-owning, for read-only access)
	struct BoundaryView
	{
		// metadata
		uint32_t N_ = 0;

		// fields
		std::span<uint32_t> id;
		std::span<uint32_t> bodyId;
		std::span<uint32_t> rigidSampleIdx;
		std::span<float> xPos, yPos, zPos;
		std::span<float> xVel, yVel, zVel;
		std::span<float> volB;
		std::span<float> psiB;
	};

	// backend-agnostic representation of fluid particles
	// (owned, for read/write access)
	struct FluidParticles
	{
		// metadata
		uint32_t N_ = 0;
		std::vector<uint32_t> id;

		// positions
		std::vector<float> xPos;
		std::vector<float> yPos;
		std::vector<float> zPos;

		// velocities
		std::vector<float> xVel;
		std::vector<float> yVel;
		std::vector<float> zVel;

		// predicted velocities
		std::vector<float> xVelPred;
		std::vector<float> yVelPred;
		std::vector<float> zVelPred;

		// pressure accelerations
		std::vector<float> xAccPress;
		std::vector<float> yAccPress;
		std::vector<float> zAccPress;

		// pressures, densities, and related solver data
		std::vector<float> sourceTerm;
		std::vector<float> diagElem;
		std::vector<float> pressure;
		std::vector<float> density;

		// reserves capacity for all vectors consistently
		void reserveAll(uint32_t N)
		{
			id.reserve(N);

			xPos.reserve(N);
			yPos.reserve(N);
			zPos.reserve(N);

			xVel.reserve(N);
			yVel.reserve(N);
			zVel.reserve(N);

			xVelPred.reserve(N);
			yVelPred.reserve(N);
			zVelPred.reserve(N);

			xAccPress.reserve(N);
			yAccPress.reserve(N);
			zAccPress.reserve(N);

			sourceTerm.reserve(N);
			diagElem.reserve(N);
			pressure.reserve(N);
			density.reserve(N);
		}

		// resizes all vectors consistently
		void resizeAll(uint32_t N)
		{
			N_ = N;

			id.resize(N_);

			xPos.resize(N_);
			yPos.resize(N_);
			zPos.resize(N_);

			xVel.resize(N_);
			yVel.resize(N_);
			zVel.resize(N_);

			xVelPred.resize(N_);
			yVelPred.resize(N_);
			zVelPred.resize(N_);

			xAccPress.resize(N_);
			yAccPress.resize(N_);
			zAccPress.resize(N_);

			sourceTerm.resize(N_);
			diagElem.resize(N_);
			pressure.resize(N_);
			density.resize(N_);
		}

		// compacts all vectors so capacity matches size
		// (useful after staged scenario construction with repeated growth)
		void shrinkToFitAll()
		{
			id.shrink_to_fit();

			xPos.shrink_to_fit();
			yPos.shrink_to_fit();
			zPos.shrink_to_fit();

			xVel.shrink_to_fit();
			yVel.shrink_to_fit();
			zVel.shrink_to_fit();

			xVelPred.shrink_to_fit();
			yVelPred.shrink_to_fit();
			zVelPred.shrink_to_fit();

			xAccPress.shrink_to_fit();
			yAccPress.shrink_to_fit();
			zAccPress.shrink_to_fit();

			sourceTerm.shrink_to_fit();
			diagElem.shrink_to_fit();
			pressure.shrink_to_fit();
			density.shrink_to_fit();
		}

		void assignSequentialIds()
		{
			id.resize(N_);
			for (uint32_t i = 0; i < N_; i++)
			{
				id[i] = i;
			}
		}

		void ensureParticleIdsPresent()
		{
			if (id.size() == static_cast<size_t>(N_)) { return; }

			assignSequentialIds();
		}

		// builds and returns a mutable view into the data
		FluidView view()
		{
			return FluidView
			{
				.N_ = N_,

				.id = std::span<uint32_t>(id),

				.xPos = std::span<float>(xPos),
				.yPos = std::span<float>(yPos),
				.zPos = std::span<float>(zPos),

				.xVel = std::span<float>(xVel),
				.yVel = std::span<float>(yVel),
				.zVel = std::span<float>(zVel),

				.xVelPred = std::span<float>(xVelPred),
				.yVelPred = std::span<float>(yVelPred),
				.zVelPred = std::span<float>(zVelPred),

				.xAccPress = std::span<float>(xAccPress),
				.yAccPress = std::span<float>(yAccPress),
				.zAccPress = std::span<float>(zAccPress),

				.sourceTerm = std::span<float>(sourceTerm),
				.diagElem = std::span<float>(diagElem),
				.pressure = std::span<float>(pressure),
				.density = std::span<float>(density)
			};
		}

		// const view reuses same layout
		FluidView constView() const
		{
			return const_cast<FluidParticles*>(this)->view();
		}

		// TODO: IISPH-specific data?
	};

	// backend-agnostic representation of boundary particles
	// (owned, for read/write access)
	struct BoundaryParticles
	{
		// metadata
		uint32_t N_ = 0;
		std::vector<uint32_t> id;
		std::vector<uint32_t> bodyId;
		std::vector<uint32_t> rigidSampleIdx;

		// positions
		std::vector<float> xPos;
		std::vector<float> yPos;
		std::vector<float> zPos;

		// velocities
		std::vector<float> xVel;
		std::vector<float> yVel;
		std::vector<float> zVel;

		// boundary contribution coefficients
		std::vector<float> volB;
		std::vector<float> psiB;

		// reserves capacity for all vectors consistently
		void reserveAll(uint32_t N)
		{
			id.reserve(N);
			bodyId.reserve(N);
			rigidSampleIdx.reserve(N);

			xPos.reserve(N);
			yPos.reserve(N);
			zPos.reserve(N);

			xVel.reserve(N);
			yVel.reserve(N);
			zVel.reserve(N);

			volB.reserve(N);
			psiB.reserve(N);
		}

		// resizes all vectors consistently
		void resizeAll(uint32_t N)
		{
			N_ = N;

			id.resize(N_);
			bodyId.resize(N_, InvalidRigidBodyId);
			rigidSampleIdx.resize(N_, InvalidRigidSampleIdx);

			xPos.resize(N_);
			yPos.resize(N_);
			zPos.resize(N_);

			xVel.resize(N_);
			yVel.resize(N_);
			zVel.resize(N_);

			volB.resize(N_);
			psiB.resize(N_);
		}

		// compacts all vectors so capacity matches size
		// (useful after staged scenario construction with repeated growth)
		void shrinkToFitAll()
		{
			id.shrink_to_fit();
			bodyId.shrink_to_fit();
			rigidSampleIdx.shrink_to_fit();

			xPos.shrink_to_fit();
			yPos.shrink_to_fit();
			zPos.shrink_to_fit();

			xVel.shrink_to_fit();
			yVel.shrink_to_fit();
			zVel.shrink_to_fit();

			volB.shrink_to_fit();
			psiB.shrink_to_fit();
		}

		void assignSequentialIds()
		{
			id.resize(N_);
			for (uint32_t i = 0; i < N_; i++)
			{
				id[i] = i;
			}
		}

		void ensureParticleIdsPresent()
		{
			if (id.size() == static_cast<size_t>(N_)) { return; }

			assignSequentialIds();
		}

		void EnsureBoundaryKinematicsPresent()
		{
			if (bodyId.size() != static_cast<uint32_t>(N_))
			{
				bodyId.assign(N_, InvalidRigidBodyId);
			}
			if (rigidSampleIdx.size() != static_cast<uint32_t>(N_))
			{
				rigidSampleIdx.assign(N_, InvalidRigidSampleIdx);
			}
			if (xVel.size() != static_cast<uint32_t>(N_))
			{
				xVel.assign(N_, 0.0f);
			}
			if (yVel.size() != static_cast<uint32_t>(N_))
			{
				yVel.assign(N_, 0.0f);
			}
			if (zVel.size() != static_cast<uint32_t>(N_))
			{
				zVel.assign(N_, 0.0f);
			}
		}

		void EnsureBoundaryCoefficientsPresent(float defaultPsiB, float restDensity)
		{
			const float defaultVolB = (restDensity > 0.0f)
				? (defaultPsiB / restDensity)
				: 0.0f;

			if (volB.size() != static_cast<uint32_t>(N_))
			{
				volB.assign(N_, defaultVolB);
			}
			if (psiB.size() != static_cast<uint32_t>(N_))
			{
				psiB.assign(N_, defaultPsiB);
			}
		}

		// builds and returns a mutable view into the data
		BoundaryView view()
		{
			return BoundaryView
			{
				.N_ = N_,

				.id = std::span<uint32_t>(id),
				.bodyId = std::span<uint32_t>(bodyId),
				.rigidSampleIdx = std::span<uint32_t>(rigidSampleIdx),
				.xPos = std::span<float>(xPos),
				.yPos = std::span<float>(yPos),
				.zPos = std::span<float>(zPos),
				.xVel = std::span<float>(xVel),
				.yVel = std::span<float>(yVel),
				.zVel = std::span<float>(zVel),
				.volB = std::span<float>(volB),
				.psiB = std::span<float>(psiB),
			};
		}

		// const view reuses same layout
		BoundaryView constView() const
		{
			return const_cast<BoundaryParticles*>(this)->view();
		}
	};

	// backend-agnostic representation of rigid bodies
	// (owned, for read/write access)
	struct RigidBodies
	{
		// metadata
		uint32_t N_ = 0;
		uint32_t sampleN_ = 0;
		std::vector<uint32_t> id; // per-rigid-body ID
		std::vector<uint32_t> sampleBegin; // first rigid boundary sample
		std::vector<uint32_t> sampleCount; // number of rigid boundary samples

		// body state (in world frame)
		std::vector<float> xCM, yCM, zCM; // center of mass
		std::vector<float> xVel, yVel, zVel; // linear velocity
		std::vector<float> xOmega, yOmega, zOmega; // angular velocity
		std::vector<float> xAngMom, yAngMom, zAngMom; // angular momentum
		std::vector<float> xForce, yForce, zForce; // force accumulated from rigid boundary samples
		std::vector<float> xTorque, yTorque, zTorque; // torque accumulated from rigid boundary samples

		// orientation matrix A, stored row-major
		std::vector<float> a00, a01, a02;
		std::vector<float> a10, a11, a12;
		std::vector<float> a20, a21, a22;

		// mass
		std::vector<float> mass;
		std::vector<float> massInv;

		// inverse inertia in body frame, stored row-major
		std::vector<float> iBodyInv00, iBodyInv01, iBodyInv02;
		std::vector<float> iBodyInv10, iBodyInv11, iBodyInv12;
		std::vector<float> iBodyInv20, iBodyInv21, iBodyInv22;

		// local rigid boundary samples offset from center of mass
		std::vector<float> xLocalB, yLocalB, zLocalB;

		void reserveAll(uint32_t N)
		{
			id.reserve(N);
			sampleBegin.reserve(N);
			sampleCount.reserve(N);

			xCM.reserve(N); yCM.reserve(N); zCM.reserve(N);
			xVel.reserve(N); yVel.reserve(N); zVel.reserve(N);
			xOmega.reserve(N); yOmega.reserve(N); zOmega.reserve(N);
			xAngMom.reserve(N); yAngMom.reserve(N); zAngMom.reserve(N);
			xForce.reserve(N); yForce.reserve(N); zForce.reserve(N);
			xTorque.reserve(N); yTorque.reserve(N); zTorque.reserve(N);

			a00.reserve(N); a01.reserve(N); a02.reserve(N);
			a10.reserve(N); a11.reserve(N); a12.reserve(N);
			a20.reserve(N); a21.reserve(N); a22.reserve(N);

			mass.reserve(N); massInv.reserve(N);

			iBodyInv00.reserve(N); iBodyInv01.reserve(N); iBodyInv02.reserve(N);
			iBodyInv10.reserve(N); iBodyInv11.reserve(N); iBodyInv12.reserve(N);
			iBodyInv20.reserve(N); iBodyInv21.reserve(N); iBodyInv22.reserve(N);
		}

		void resizeAll(uint32_t N)
		{
			N_ = N;

			id.resize(N_);
			sampleBegin.resize(N_);
			sampleCount.resize(N_);

			xCM.resize(N_); yCM.resize(N_); zCM.resize(N_);
			xVel.resize(N_); yVel.resize(N_); zVel.resize(N_);
			xOmega.resize(N_); yOmega.resize(N_); zOmega.resize(N_);
			xAngMom.resize(N_); yAngMom.resize(N_); zAngMom.resize(N_);
			xForce.resize(N_); yForce.resize(N_); zForce.resize(N_);
			xTorque.resize(N_); yTorque.resize(N_); zTorque.resize(N_);

			a00.resize(N_, 1.0f); a01.resize(N_); a02.resize(N_);
			a10.resize(N_); a11.resize(N_, 1.0f); a12.resize(N_);
			a20.resize(N_); a21.resize(N_); a22.resize(N_, 1.0f);

			mass.resize(N_);
			massInv.resize(N_);

			iBodyInv00.resize(N_); iBodyInv01.resize(N_); iBodyInv02.resize(N_);
			iBodyInv10.resize(N_); iBodyInv11.resize(N_); iBodyInv12.resize(N_);
			iBodyInv20.resize(N_); iBodyInv21.resize(N_); iBodyInv22.resize(N_);
		}

		void reserveSamples(uint32_t N)
		{
			xLocalB.reserve(N);
			yLocalB.reserve(N);
			zLocalB.reserve(N);
		}

		void resizeSamples(uint32_t N)
		{
			sampleN_ = N;
			xLocalB.resize(sampleN_);
			yLocalB.resize(sampleN_);
			zLocalB.resize(sampleN_);
		}

		void shrinkToFitAll()
		{
			id.shrink_to_fit();
			sampleBegin.shrink_to_fit();
			sampleCount.shrink_to_fit();

			xCM.shrink_to_fit(); yCM.shrink_to_fit(); zCM.shrink_to_fit();
			xVel.shrink_to_fit(); yVel.shrink_to_fit(); zVel.shrink_to_fit();
			xOmega.shrink_to_fit(); yOmega.shrink_to_fit(); zOmega.shrink_to_fit();
			xAngMom.shrink_to_fit(); yAngMom.shrink_to_fit(); zAngMom.shrink_to_fit();
			xForce.shrink_to_fit(); yForce.shrink_to_fit(); zForce.shrink_to_fit();
			xTorque.shrink_to_fit(); yTorque.shrink_to_fit(); zTorque.shrink_to_fit();

			a00.shrink_to_fit(); a01.shrink_to_fit(); a02.shrink_to_fit();
			a10.shrink_to_fit(); a11.shrink_to_fit(); a12.shrink_to_fit();
			a20.shrink_to_fit(); a21.shrink_to_fit(); a22.shrink_to_fit();

			mass.shrink_to_fit(); massInv.shrink_to_fit();

			iBodyInv00.shrink_to_fit(); iBodyInv01.shrink_to_fit(); iBodyInv02.shrink_to_fit();
			iBodyInv10.shrink_to_fit(); iBodyInv11.shrink_to_fit(); iBodyInv12.shrink_to_fit();
			iBodyInv20.shrink_to_fit(); iBodyInv21.shrink_to_fit(); iBodyInv22.shrink_to_fit();

			xLocalB.shrink_to_fit();
			yLocalB.shrink_to_fit();
			zLocalB.shrink_to_fit();
		}

		void assignSequentialIds()
		{
			id.resize(N_);

			for (uint32_t i = 0; i < N_; i++)
			{
				id[i] = i;
			}
		}

		void ensureRigidBodyIdsPresent()
		{
			if (id.size() == static_cast<size_t>(N_)) { return; }

			assignSequentialIds();
		}
	};

	// wrapper of backend-agnostic data structure for capturing the full simulation state,
	// including all data making up the state, enabling lossless save & load
	// (mostly SoA-layout for efficient serialization or conversion into other layouts)
	// TODO: how to deal with simulation method-specific data?
	struct UniversalSimState
	{
		// simulation metadata
		uint32_t step = 0;			// simulation step at this state
		double time = 0.0;			// simulation time in seconds

		// particle data
		FluidParticles fluidParticles;
		BoundaryParticles boundaryParticles;
		RigidBodies rigidBodies;

		// convenience methods for reserve/resizing
		void ReserveFluid(uint32_t N) { fluidParticles.reserveAll(N); }
		void ReserveBoundary(uint32_t N) { boundaryParticles.reserveAll(N); }
		void ReserveRigidBodies(uint32_t N) { rigidBodies.reserveAll(N); }
		void ResizeFluid(uint32_t N) { fluidParticles.resizeAll(N); }
		void ResizeBoundary(uint32_t N) { boundaryParticles.resizeAll(N); }
		void ResizeRigidBodies(uint32_t N) { rigidBodies.resizeAll(N); }
		void ShrinkToFitAll()
		{
			fluidParticles.shrinkToFitAll();
			boundaryParticles.shrinkToFitAll();
			rigidBodies.shrinkToFitAll();
		}

		void ReserveFluidAdditional(uint32_t add)
		{
			const auto current = static_cast<uint32_t>(fluidParticles.xPos.size());
			fluidParticles.reserveAll(current + add);
		}

		void ReserveBoundaryAdditional(uint32_t add)
		{
			const auto current = static_cast<uint32_t>(boundaryParticles.xPos.size());
			boundaryParticles.reserveAll(current + add);
		}

		// getters for views into the particle data
		// NOTE: do not call resize() while holding a view
		FluidView GetFluidView() { return fluidParticles.view(); }
		BoundaryView GetBoundaryView() { return boundaryParticles.view(); }

		// TODO: add snapshot of used simulation parameters?
	};

} // namespace Sync
