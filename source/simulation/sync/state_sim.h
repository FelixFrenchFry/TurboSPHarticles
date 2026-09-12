// state_sim.h
#pragma once

// external
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

// internal
#include "source/simulation/core/particle_types.h"



namespace Sync
{
	inline constexpr uint32_t InvalidRigidBodyId = std::numeric_limits<uint32_t>::max();

	// view into fluid particle data
	// (non-owning, for read-only access)
	struct FluidConstView
	{
		// metadata
		uint32_t N_ = 0;

		// fields
		std::span<const uint64_t> id;
		std::span<const float> xPos, yPos, zPos;
		std::span<const float> xVel, yVel, zVel;
		std::span<const float> xVelPred, yVelPred, zVelPred;
		std::span<const float> xAccPress, yAccPress, zAccPress;
		std::span<const float> sourceTerm;
		std::span<const float> diagElem;
		std::span<const float> pressure;
		std::span<const float> density;
	};

	// view into boundary particle data
	// (non-owning, for read-only access)
	struct BoundaryConstView
	{
		// metadata
		uint32_t N_ = 0;

		// fields
		std::span<const uint64_t> id;
		std::span<const float> xPos, yPos, zPos;
		std::span<const float> volB;
		std::span<const float> psiB;
	};

	// view into rigid particle data
	// (non-owning, for read-only access)
	struct RigidConstView
	{
		// metadata
		uint32_t N_ = 0;

		// fields
		std::span<const uint64_t> id;
		std::span<const uint32_t> bodyId;
		std::span<const float> xRelPos, yRelPos, zRelPos;
		std::span<const float> volR;
		std::span<const float> psiR;
	};

	// backend-agnostic representation of fluid particles
	// (owned, for read/write access)
	struct FluidParticles
	{
		// metadata
		uint32_t N_ = 0;
		std::vector<uint64_t> id;

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
			(void)Sim::CheckedLocalParticleCount(N, "Fluid reserve");
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
			(void)Sim::CheckedLocalParticleCount(N, "Fluid resize");
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

		// builds and returns an immutable view into the data
		FluidConstView constView() const
		{
			return FluidConstView
			{
				.N_ = N_,

				.id = std::span<const uint64_t>(id),

				.xPos = std::span<const float>(xPos),
				.yPos = std::span<const float>(yPos),
				.zPos = std::span<const float>(zPos),

				.xVel = std::span<const float>(xVel),
				.yVel = std::span<const float>(yVel),
				.zVel = std::span<const float>(zVel),

				.xVelPred = std::span<const float>(xVelPred),
				.yVelPred = std::span<const float>(yVelPred),
				.zVelPred = std::span<const float>(zVelPred),

				.xAccPress = std::span<const float>(xAccPress),
				.yAccPress = std::span<const float>(yAccPress),
				.zAccPress = std::span<const float>(zAccPress),

				.sourceTerm = std::span<const float>(sourceTerm),
				.diagElem = std::span<const float>(diagElem),
				.pressure = std::span<const float>(pressure),
				.density = std::span<const float>(density)
			};
		}

		// TODO: IISPH-specific data?
	};

	// backend-agnostic representation of boundary particles
	// (owned, for read/write access)
	struct BoundaryParticles
	{
		// metadata
		uint32_t N_ = 0;
		std::vector<uint64_t> id;

		// positions
		std::vector<float> xPos;
		std::vector<float> yPos;
		std::vector<float> zPos;

		// boundary contribution coefficients
		std::vector<float> volB;
		std::vector<float> psiB;

		// reserves capacity for all vectors consistently
		void reserveAll(uint32_t N)
		{
			(void)Sim::CheckedLocalParticleCount(N, "Boundary reserve");
			id.reserve(N);

			xPos.reserve(N);
			yPos.reserve(N);
			zPos.reserve(N);

			volB.reserve(N);
			psiB.reserve(N);
		}

		// resizes all vectors consistently
		void resizeAll(uint32_t N)
		{
			(void)Sim::CheckedLocalParticleCount(N, "Boundary resize");
			N_ = N;

			id.resize(N_);

			xPos.resize(N_);
			yPos.resize(N_);
			zPos.resize(N_);

			volB.resize(N_);
			psiB.resize(N_);
		}

		// compacts all vectors so capacity matches size
		// (useful after staged scenario construction with repeated growth)
		void shrinkToFitAll()
		{
			id.shrink_to_fit();

			xPos.shrink_to_fit();
			yPos.shrink_to_fit();
			zPos.shrink_to_fit();

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

		void ensureBoundaryCoefficientsPresent(float defaultPsiB, float restDensity)
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

		// builds and returns an immutable view into the data
		BoundaryConstView constView() const
		{
			return BoundaryConstView
			{
				.N_ = N_,

				.id = std::span<const uint64_t>(id),
				.xPos = std::span<const float>(xPos),
				.yPos = std::span<const float>(yPos),
				.zPos = std::span<const float>(zPos),
				.volB = std::span<const float>(volB),
				.psiB = std::span<const float>(psiB),
			};
		}
	};

	// backend-agnostic representation of rigid particles
	// (owned, for read/write access)
	struct RigidParticles
	{
		// metadata
		uint32_t N_ = 0;
		std::vector<uint64_t> id;
		std::vector<uint32_t> bodyId;

		// positions relative to the rigid body center of mass
		std::vector<float> xRelPos;
		std::vector<float> yRelPos;
		std::vector<float> zRelPos;

		// boundary contribution coefficients
		std::vector<float> volR;
		std::vector<float> psiR;

		// reserves capacity for all vectors consistently
		void reserveAll(uint32_t N)
		{
			(void)Sim::CheckedLocalParticleCount(N, "Rigid reserve");
			id.reserve(N);
			bodyId.reserve(N);

			xRelPos.reserve(N);
			yRelPos.reserve(N);
			zRelPos.reserve(N);

			volR.reserve(N);
			psiR.reserve(N);
		}

		// resizes all vectors consistently
		void resizeAll(uint32_t N)
		{
			(void)Sim::CheckedLocalParticleCount(N, "Rigid resize");
			N_ = N;

			id.resize(N_);
			bodyId.resize(N_, InvalidRigidBodyId);

			xRelPos.resize(N_);
			yRelPos.resize(N_);
			zRelPos.resize(N_);

			volR.resize(N_);
			psiR.resize(N_);
		}

		// compacts all vectors so capacity matches size
		// (useful after staged scenario construction with repeated growth)
		void shrinkToFitAll()
		{
			id.shrink_to_fit();
			bodyId.shrink_to_fit();

			xRelPos.shrink_to_fit();
			yRelPos.shrink_to_fit();
			zRelPos.shrink_to_fit();

			volR.shrink_to_fit();
			psiR.shrink_to_fit();
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

		void ensureRigidCoefficientsPresent(float defaultPsiR, float restDensity)
		{
			const float defaultVolR = (restDensity > 0.0f)
				? (defaultPsiR / restDensity)
				: 0.0f;

			if (volR.size() != static_cast<uint32_t>(N_))
			{
				volR.assign(N_, defaultVolR);
			}
			if (psiR.size() != static_cast<uint32_t>(N_))
			{
				psiR.assign(N_, defaultPsiR);
			}
		}

		// builds and returns an immutable view into the data
		RigidConstView constView() const
		{
			return RigidConstView
			{
				.N_ = N_,

				.id = std::span<const uint64_t>(id),
				.bodyId = std::span<const uint32_t>(bodyId),
				.xRelPos = std::span<const float>(xRelPos),
				.yRelPos = std::span<const float>(yRelPos),
				.zRelPos = std::span<const float>(zRelPos),
				.volR = std::span<const float>(volR),
				.psiR = std::span<const float>(psiR),
			};
		}
	};

	// backend-agnostic representation of rigid bodies
	// (owned, for read/write access)
	struct RigidBodies
	{
		// metadata
		uint32_t N_ = 0;
		std::vector<uint32_t> id; // per-rigid-body ID
		std::vector<uint32_t> rigidParticleBegin; // first rigid particle
		std::vector<uint32_t> rigidParticleCount; // number of rigid particles

		// body state (in world frame)
		std::vector<float> xCM, yCM, zCM; // center of mass
		std::vector<float> xVel, yVel, zVel; // linear velocity
		std::vector<float> xOmega, yOmega, zOmega; // angular velocity
		std::vector<float> xAngMom, yAngMom, zAngMom; // angular momentum

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

		void reserveAll(uint32_t N)
		{
			id.reserve(N);
			rigidParticleBegin.reserve(N);
			rigidParticleCount.reserve(N);

			xCM.reserve(N); yCM.reserve(N); zCM.reserve(N);
			xVel.reserve(N); yVel.reserve(N); zVel.reserve(N);
			xOmega.reserve(N); yOmega.reserve(N); zOmega.reserve(N);
			xAngMom.reserve(N); yAngMom.reserve(N); zAngMom.reserve(N);

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
			rigidParticleBegin.resize(N_);
			rigidParticleCount.resize(N_);

			xCM.resize(N_); yCM.resize(N_); zCM.resize(N_);
			xVel.resize(N_); yVel.resize(N_); zVel.resize(N_);
			xOmega.resize(N_); yOmega.resize(N_); zOmega.resize(N_);
			xAngMom.resize(N_); yAngMom.resize(N_); zAngMom.resize(N_);

			a00.resize(N_, 1.0f); a01.resize(N_); a02.resize(N_);
			a10.resize(N_); a11.resize(N_, 1.0f); a12.resize(N_);
			a20.resize(N_); a21.resize(N_); a22.resize(N_, 1.0f);

			mass.resize(N_);
			massInv.resize(N_);

			iBodyInv00.resize(N_); iBodyInv01.resize(N_); iBodyInv02.resize(N_);
			iBodyInv10.resize(N_); iBodyInv11.resize(N_); iBodyInv12.resize(N_);
			iBodyInv20.resize(N_); iBodyInv21.resize(N_); iBodyInv22.resize(N_);
		}

		void shrinkToFitAll()
		{
			id.shrink_to_fit();
			rigidParticleBegin.shrink_to_fit();
			rigidParticleCount.shrink_to_fit();

			xCM.shrink_to_fit(); yCM.shrink_to_fit(); zCM.shrink_to_fit();
			xVel.shrink_to_fit(); yVel.shrink_to_fit(); zVel.shrink_to_fit();
			xOmega.shrink_to_fit(); yOmega.shrink_to_fit(); zOmega.shrink_to_fit();
			xAngMom.shrink_to_fit(); yAngMom.shrink_to_fit(); zAngMom.shrink_to_fit();

			a00.shrink_to_fit(); a01.shrink_to_fit(); a02.shrink_to_fit();
			a10.shrink_to_fit(); a11.shrink_to_fit(); a12.shrink_to_fit();
			a20.shrink_to_fit(); a21.shrink_to_fit(); a22.shrink_to_fit();

			mass.shrink_to_fit(); massInv.shrink_to_fit();

			iBodyInv00.shrink_to_fit(); iBodyInv01.shrink_to_fit(); iBodyInv02.shrink_to_fit();
			iBodyInv10.shrink_to_fit(); iBodyInv11.shrink_to_fit(); iBodyInv12.shrink_to_fit();
			iBodyInv20.shrink_to_fit(); iBodyInv21.shrink_to_fit(); iBodyInv22.shrink_to_fit();
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
		uint64_t step = 0;			// simulation step at this state
		double time = 0.0;			// simulation time in seconds

		// particle data
		FluidParticles particlesF;
		BoundaryParticles particlesB;
		RigidParticles particlesR;
		RigidBodies rigidBodies;

		// convenience methods for reserve/resizing
		void ReserveFluid(uint64_t N)
		{
			particlesF.reserveAll(Sim::CheckedLocalParticleCount(N, "Fluid reserve"));
		}
		void ReserveBoundary(uint64_t N)
		{
			particlesB.reserveAll(Sim::CheckedLocalParticleCount(N, "Boundary reserve"));
		}
		void ReserveRigid(uint64_t N)
		{
			particlesR.reserveAll(Sim::CheckedLocalParticleCount(N, "Rigid reserve"));
		}
		void ReserveRigidBodies(uint32_t N) { rigidBodies.reserveAll(N); }
		void ResizeFluid(uint64_t N)
		{
			particlesF.resizeAll(Sim::CheckedLocalParticleCount(N, "Fluid resize"));
		}
		void ResizeBoundary(uint64_t N)
		{
			particlesB.resizeAll(Sim::CheckedLocalParticleCount(N, "Boundary resize"));
		}
		void ResizeRigid(uint64_t N)
		{
			particlesR.resizeAll(Sim::CheckedLocalParticleCount(N, "Rigid resize"));
		}
		void ResizeRigidBodies(uint32_t N) { rigidBodies.resizeAll(N); }
		void ShrinkToFitAll()
		{
			particlesF.shrinkToFitAll();
			particlesB.shrinkToFitAll();
			particlesR.shrinkToFitAll();
			rigidBodies.shrinkToFitAll();
		}

		void ReserveFluidAdditional(uint64_t add)
		{
			const uint64_t total = Sim::CheckedParticleAdd(
				particlesF.xPos.size(), add, "Fluid reserve");
			particlesF.reserveAll(Sim::CheckedLocalParticleCount(total, "Fluid reserve"));
		}

		void ReserveBoundaryAdditional(uint64_t add)
		{
			const uint64_t total = Sim::CheckedParticleAdd(
				particlesB.xPos.size(), add, "Boundary reserve");
			particlesB.reserveAll(Sim::CheckedLocalParticleCount(total, "Boundary reserve"));
		}

		void ReserveRigidAdditional(uint64_t add)
		{
			const uint64_t total = Sim::CheckedParticleAdd(
				particlesR.xRelPos.size(), add, "Rigid reserve");
			particlesR.reserveAll(Sim::CheckedLocalParticleCount(total, "Rigid reserve"));
		}

		// validates rigid particle ownership before backend hot loops perform updates
		void ValidateRigidAttachments() const
		{
			const auto& R = particlesR;
			const auto& bodies = rigidBodies;

			if (R.id.size() != static_cast<size_t>(R.N_) ||
				R.bodyId.size() != static_cast<size_t>(R.N_) ||
				R.xRelPos.size() != static_cast<size_t>(R.N_) ||
				R.yRelPos.size() != static_cast<size_t>(R.N_) ||
				R.zRelPos.size() != static_cast<size_t>(R.N_) ||
				R.volR.size() != static_cast<size_t>(R.N_) ||
				R.psiR.size() != static_cast<size_t>(R.N_))
			{
				throw std::runtime_error("Invalid rigid particle array size.");
			}

			if (bodies.rigidParticleBegin.size() != static_cast<size_t>(bodies.N_) ||
				bodies.rigidParticleCount.size() != static_cast<size_t>(bodies.N_))
			{
				throw std::runtime_error("Invalid rigid body particle range array size.");
			}

			uint32_t nextRigidParticleIdx = 0;
			for (uint32_t bodyIdx = 0; bodyIdx < bodies.N_; bodyIdx++)
			{
				const uint32_t begin = bodies.rigidParticleBegin[bodyIdx];
				const uint32_t N_rigidParticles = bodies.rigidParticleCount[bodyIdx];
				const uint32_t end = begin + N_rigidParticles;

				if (begin != nextRigidParticleIdx)
				{
					throw std::runtime_error("Rigid body particle ranges are not contiguous.");
				}
				if (begin > R.N_ || N_rigidParticles > R.N_ - begin)
				{
					throw std::runtime_error("Invalid rigid body particle range.");
				}

				for (uint32_t rigidParticleIdx = begin; rigidParticleIdx < end; rigidParticleIdx++)
				{
					if (R.bodyId[rigidParticleIdx] != bodyIdx)
					{
						throw std::runtime_error("Rigid particle belongs to a different body.");
					}
				}

				nextRigidParticleIdx = end;
			}

			if (nextRigidParticleIdx != R.N_)
			{
				throw std::runtime_error("Rigid body particle ranges do not cover all rigid particles.");
			}
		}

		// TODO: add snapshot of used simulation parameters?
	};

} // namespace Sync
