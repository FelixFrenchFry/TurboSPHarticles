// simulation_state.h
//
// TODO: comment

#pragma once

// external
#include <cstdint>
#include <span>
#include <vector>

// internal



namespace Sync
{
	// view into fluid particle data
	// (non-owning, for read-only access)
	struct FluidView
	{
		// metadata
		std::int64_t N_particles{ 0 };
		std::span<std::uint32_t> id; // unique particle IDs

		// fields
		std::span<float> xPos, yPos, zPos;
		std::span<float> xVel, yVel, zVel;
		std::span<float> xAcc, yAcc, zAcc;
		std::span<float> xAccPress, yAccPress, zAccPress;
		std::span<float> xAccVisc, yAccVisc, zAccVisc;
		std::span<float> xAccExt, yAccExt, zAccExt;
		std::span<float> density;
		std::span<float> pressure;
		std::span<float> mass;
		std::span<float> size;
	};

	// view into boundary particle data
	// (non-owning, for read-only access)
	struct BoundaryView
	{
		// metadata
		std::int64_t N_particles{ 0 };
		std::span<std::uint32_t> id; // unique particle IDs

		// fields
		std::span<float> xPos, yPos, zPos;
		std::span<float> xVel, yVel, zVel;
		std::span<float> density;
		std::span<float> pressure;
		std::span<float> mass;
		std::span<float> size;
	};

	// backend-agnostic representation of fluid particles
	// (owned, for read/write access)
	struct FluidParticles
	{
		// metadata
		std::int64_t N_particles{ 0 };
		std::vector<std::uint32_t> id; // unique particle IDs

		// positions
		std::vector<float> xPos;
		std::vector<float> yPos;
		std::vector<float> zPos;

		// velocities
		std::vector<float> xVel;
		std::vector<float> yVel;
		std::vector<float> zVel;

		// accelerations (total)
		std::vector<float> xAcc;
		std::vector<float> yAcc;
		std::vector<float> zAcc;

		// pressure accelerations
		std::vector<float> xAccPress;
		std::vector<float> yAccPress;
		std::vector<float> zAccPress;

		// viscosity accelerations
		std::vector<float> xAccVisc;
		std::vector<float> yAccVisc;
		std::vector<float> zAccVisc;

		// external accelerations
		// (gravity, surface tensions, etc...)
		std::vector<float> xAccExt;
		std::vector<float> yAccExt;
		std::vector<float> zAccExt;

		// densities and pressures
		std::vector<float> density;
		std::vector<float> pressure;

		// masses and sizes
		std::vector<float> mass;
		std::vector<float> size;

		// resizes all vectors consistently
		void resize(std::uint32_t N)
		{
			N_particles = N;
			id.resize(N_particles);

			xPos.resize(N_particles);
			yPos.resize(N_particles);
			zPos.resize(N_particles);

			xVel.resize(N_particles);
			yVel.resize(N_particles);
			zVel.resize(N_particles);

			xAcc.resize(N_particles);
			yAcc.resize(N_particles);
			zAcc.resize(N_particles);

			xAccPress.resize(N_particles);
			yAccPress.resize(N_particles);
			zAccPress.resize(N_particles);

			xAccVisc.resize(N_particles);
			yAccVisc.resize(N_particles);
			zAccVisc.resize(N_particles);

			xAccExt.resize(N_particles);
			yAccExt.resize(N_particles);
			zAccExt.resize(N_particles);

			density.resize(N_particles);
			pressure.resize(N_particles);
			mass.resize(N_particles);
			size.resize(N_particles);
		}

		// builds and returns a mutable view into the data
		FluidView view()
		{
			return FluidView
			{
				.N_particles = N_particles,
				.id = std::span<std::uint32_t>(id),

				.xPos = std::span<float>(xPos),
				.yPos = std::span<float>(yPos),
				.zPos = std::span<float>(zPos),

				.xVel = std::span<float>(xVel),
				.yVel = std::span<float>(yVel),
				.zVel = std::span<float>(zVel),

				.xAcc = std::span<float>(xAcc),
				.yAcc = std::span<float>(yAcc),
				.zAcc = std::span<float>(zAcc),

				.xAccPress = std::span<float>(xAccPress),
				.yAccPress = std::span<float>(yAccPress),
				.zAccPress = std::span<float>(zAccPress),

				.xAccVisc = std::span<float>(xAccVisc),
				.yAccVisc = std::span<float>(yAccVisc),
				.zAccVisc = std::span<float>(zAccVisc),

				.xAccExt = std::span<float>(xAccExt),
				.yAccExt = std::span<float>(yAccExt),
				.zAccExt = std::span<float>(zAccExt),

				.density = std::span<float>(density),
				.pressure = std::span<float>(pressure),
				.mass = std::span<float>(mass),
				.size = std::span<float>(size)
			};
		}

		// const view reuses same layout
		FluidView constView() const
		{
			return const_cast<FluidParticles*>(this)->view();
		}

		// TODO: SESPH-specific data?

		// TODO: IISPH-specific data?
	};

	// backend-agnostic representation of boundary particles
	// (owned, for read/write access)
	struct BoundaryParticles
	{
		// metadata
		std::int64_t N_particles{ 0 };
		std::vector<std::uint32_t> id; // unique particle IDs

		// positions
		std::vector<float> xPos;
		std::vector<float> yPos;
		std::vector<float> zPos;

		// velocities
		std::vector<float> xVel;
		std::vector<float> yVel;
		std::vector<float> zVel;

		// TODO: accelerations for rigid moving boundaries?

		// densities and pressures
		std::vector<float> density;
		std::vector<float> pressure;

		// masses and sizes
		std::vector<float> mass;
		std::vector<float> size;

		// resizes all vectors consistently
		void resize(std::int64_t N)
		{
			N_particles = N;
			id.resize(N_particles);

			xPos.resize(N_particles);
			yPos.resize(N_particles);
			zPos.resize(N_particles);

			xVel.resize(N_particles);
			yVel.resize(N_particles);
			zVel.resize(N_particles);

			density.resize(N_particles);
			pressure.resize(N_particles);
			mass.resize(N_particles);
			size.resize(N_particles);
		}

		// builds and returns a mutable view into the data
		BoundaryView view()
		{
			return BoundaryView
			{
				.N_particles = N_particles,
				.id = std::span<std::uint32_t>(id),

				.xPos = std::span<float>(xPos),
				.yPos = std::span<float>(yPos),
				.zPos = std::span<float>(zPos),

				.xVel = std::span<float>(xVel),
				.yVel = std::span<float>(yVel),
				.zVel = std::span<float>(zVel),

				.density = std::span<float>(density),
				.pressure = std::span<float>(pressure),
				.mass = std::span<float>(mass),
				.size = std::span<float>(size)
			};
		}

		// const view reuses same layout
		BoundaryView constView() const
		{
			return const_cast<BoundaryParticles*>(this)->view();
		}

		// TODO: SESPH-specific data?

		// TODO: IISPH-specific data?
	};

	// wrapper of backend-agnostic data structure for capturing the full simulation state,
	// including all data making up the state, enabling lossless save & load
	// (mostly SoA-layout for efficient serialization or conversion into other layouts)
	// TODO: how to deal with simulation method-specific data?
	// TODO: how to deal with backend-specific data?
	struct UniversalSimState
	{
		// simulation metadata
		std::int64_t step{ 0 };		// simulation step at this state
		double time{ 0 };			// simulation time in seconds

		// persistent layout version
		std::int64_t schemaVersion{ 1 };

		// particle data
		FluidParticles fluidParticles;
		BoundaryParticles boundaryParticles;

		// convenience methods for resizing the particle arrays
		void ResizeFluid(std::int64_t N) { fluidParticles.resize(N); }
		void ResizeBoundary(std::int64_t N) { boundaryParticles.resize(N); }

		// getters for views into the particle data
		// NOTE: do not call resize() while holding a view
		FluidView GetFluidView() { return fluidParticles.view(); }
		BoundaryView GetBoundaryView() { return boundaryParticles.view(); }

		// TODO: add snapshot of used simulation parameters?
	};

} // namespace Sync
