// state_sim.h
//
// TODO: comment

#pragma once

// external
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
		std::span<float> xPos, yPos, zPos;
		std::span<float> xVel, yVel, zVel;
		std::span<float> xVelPred, yVelPred, zVelPred;
		std::span<float> xAccPress, yAccPress, zAccPress;
		std::span<float> xAccNonPress, yAccNonPress, zAccNonPress;
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
		std::span<float> xPos, yPos, zPos;
	};

	// backend-agnostic representation of fluid particles
	// (owned, for read/write access)
	struct FluidParticles
	{
		// metadata
		uint32_t N_ = 0;

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

		// non-pressure accelerations
		// (viscosity, surface tension, gravity, etc...)
		std::vector<float> xAccNonPress;
		std::vector<float> yAccNonPress;
		std::vector<float> zAccNonPress;

		// pressures, densities, and related solver data
		std::vector<float> sourceTerm;
		std::vector<float> diagElem;
		std::vector<float> pressure;
		std::vector<float> density;

		// reserves capacity for all vectors consistently
		void reserveAll(uint32_t N)
		{
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

			xAccNonPress.reserve(N);
			yAccNonPress.reserve(N);
			zAccNonPress.reserve(N);

			sourceTerm.reserve(N);
			diagElem.reserve(N);
			pressure.reserve(N);
			density.reserve(N);
		}

		// resizes all vectors consistently
		void resizeAll(uint32_t N)
		{
			N_ = N;

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

			xAccNonPress.resize(N_);
			yAccNonPress.resize(N_);
			zAccNonPress.resize(N_);

			sourceTerm.resize(N_);
			diagElem.resize(N_);
			pressure.resize(N_);
			density.resize(N_);
		}

		// builds and returns a mutable view into the data
		FluidView view()
		{
			return FluidView
			{
				.N_ = N_,

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

				.xAccNonPress = std::span<float>(xAccNonPress),
				.yAccNonPress = std::span<float>(yAccNonPress),
				.zAccNonPress = std::span<float>(zAccNonPress),

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

		// positions
		std::vector<float> xPos;
		std::vector<float> yPos;
		std::vector<float> zPos;

		// reserves capacity for all vectors consistently
		void reserveAll(uint32_t N)
		{
			xPos.reserve(N);
			yPos.reserve(N);
			zPos.reserve(N);
		}

		// resizes all vectors consistently
		void resizeAll(uint32_t N)
		{
			N_ = N;

			xPos.resize(N_);
			yPos.resize(N_);
			zPos.resize(N_);
		}

		// builds and returns a mutable view into the data
		BoundaryView view()
		{
			return BoundaryView
			{
				.N_ = N_,

				.xPos = std::span<float>(xPos),
				.yPos = std::span<float>(yPos),
				.zPos = std::span<float>(zPos),
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
		// persistent layout version
		uint64_t schemaVersion = 1;

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
