// state_sim.h
//
// TODO: comment

#pragma once

// external
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>



namespace Sync
{
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
		std::span<float> xPos, yPos, zPos;
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

		void EnsureBoundaryCoefficientsPresent(float defaultPsiB, float restDensityF)
		{
			const float defaultVolB = (restDensityF > 0.0f)
				? (defaultPsiB / restDensityF)
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
				.xPos = std::span<float>(xPos),
				.yPos = std::span<float>(yPos),
				.zPos = std::span<float>(zPos),
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

		// convenience methods for reserve/resizing
		void ReserveFluid(uint32_t N) { fluidParticles.reserveAll(N); }
		void ReserveBoundary(uint32_t N) { boundaryParticles.reserveAll(N); }
		void ResizeFluid(uint32_t N) { fluidParticles.resizeAll(N); }
		void ResizeBoundary(uint32_t N) { boundaryParticles.resizeAll(N); }
		void ShrinkToFitAll()
		{
			fluidParticles.shrinkToFitAll();
			boundaryParticles.shrinkToFitAll();
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
