// state_serialization.cpp

// external
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <iterator>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <system_error>
#include <type_traits>
#include <vector>

// internal
#include "source/eval/serial/serialization.h"



namespace
{
	// independent field lists keep the tests separate from the production serialization visitors
	constexpr std::array fluidFields{
		&Sync::FluidParticles::xPos, &Sync::FluidParticles::yPos, &Sync::FluidParticles::zPos,
		&Sync::FluidParticles::xVel, &Sync::FluidParticles::yVel, &Sync::FluidParticles::zVel,
		&Sync::FluidParticles::xVelPred, &Sync::FluidParticles::yVelPred, &Sync::FluidParticles::zVelPred,
		&Sync::FluidParticles::xAccPress, &Sync::FluidParticles::yAccPress, &Sync::FluidParticles::zAccPress,
		&Sync::FluidParticles::sourceTerm, &Sync::FluidParticles::diagElem,
		&Sync::FluidParticles::pressure, &Sync::FluidParticles::density };
	constexpr std::array boundaryFields{
		&Sync::BoundaryParticles::xPos, &Sync::BoundaryParticles::yPos, &Sync::BoundaryParticles::zPos,
		&Sync::BoundaryParticles::volB, &Sync::BoundaryParticles::psiB };
	constexpr std::array rigidFields{
		&Sync::RigidParticles::xPosRel, &Sync::RigidParticles::yPosRel, &Sync::RigidParticles::zPosRel,
		&Sync::RigidParticles::volR, &Sync::RigidParticles::psiR };
	constexpr std::array bodyFields{
		&Sync::RigidBodies::xCM, &Sync::RigidBodies::yCM, &Sync::RigidBodies::zCM,
		&Sync::RigidBodies::xVel, &Sync::RigidBodies::yVel, &Sync::RigidBodies::zVel,
		&Sync::RigidBodies::xOmega, &Sync::RigidBodies::yOmega, &Sync::RigidBodies::zOmega,
		&Sync::RigidBodies::xAngMom, &Sync::RigidBodies::yAngMom, &Sync::RigidBodies::zAngMom,
		&Sync::RigidBodies::a00, &Sync::RigidBodies::a01, &Sync::RigidBodies::a02,
		&Sync::RigidBodies::a10, &Sync::RigidBodies::a11, &Sync::RigidBodies::a12,
		&Sync::RigidBodies::a20, &Sync::RigidBodies::a21, &Sync::RigidBodies::a22,
		&Sync::RigidBodies::mass, &Sync::RigidBodies::massInv,
		&Sync::RigidBodies::iBodyInv00, &Sync::RigidBodies::iBodyInv01, &Sync::RigidBodies::iBodyInv02,
		&Sync::RigidBodies::iBodyInv10, &Sync::RigidBodies::iBodyInv11, &Sync::RigidBodies::iBodyInv12,
		&Sync::RigidBodies::iBodyInv20, &Sync::RigidBodies::iBodyInv21, &Sync::RigidBodies::iBodyInv22 };

	template <class Particles, std::size_t N_fields>
	void FillFields(Particles& particles, const std::array<std::vector<float> Particles::*, N_fields>& fields,
		const float base)
	{
		for (uint32_t field = 0; field < fields.size(); field++)
		{
			auto& values = particles.*fields[field];
			for (uint32_t i = 0; i < values.size(); i++)
			{
				values[i] = base + static_cast<float>(16 * field + i) + 0.25f;
			}
		}
	}

	Sync::UniversalSimState MakeState(const uint32_t N_F = 2, const uint32_t N_B = 1,
		const uint32_t N_R = 3, const uint32_t N_RB = 2)
	{
		Sync::UniversalSimState state;
		state.step = 0x0102030405060708ull;
		state.time = 12.5;
		state.ResizeFluid(N_F);
		state.ResizeBoundary(N_B);
		state.ResizeRigid(N_R);
		state.ResizeRigidBodies(N_RB);
		FillFields(state.particlesF, fluidFields, 100.0f);
		FillFields(state.particlesB, boundaryFields, 1000.0f);
		FillFields(state.particlesR, rigidFields, 2000.0f);
		FillFields(state.rigidBodies, bodyFields, 3000.0f);
		for (uint32_t i = 0; i < N_F; i++) { state.particlesF.id[i] = 5'000'000'000ull + 17 * i; }
		for (uint32_t i = 0; i < N_B; i++) { state.particlesB.id[i] = 6'000'000'000ull + 19 * i; }
		for (uint32_t i = 0; i < N_RB; i++) { state.rigidBodies.id[i] = 100 + 7 * i; }
		for (uint32_t i = 0; i < N_R; i++)
		{
			state.particlesR.id[i] = 7'000'000'000ull + 23 * i;
			state.particlesR.bodyId[i] = i % N_RB;
			state.rigidBodies.rigidParticleCount[i % N_RB]++;
		}
		if (N_F > 0)
		{
			// signed zero and the largest valid particle ID must survive without losing bits
			state.particlesF.id[0] = std::numeric_limits<uint64_t>::max() - 1;
			state.particlesF.yVel[0] = -123.5f;
			state.particlesF.zVel[0] = -0.0f;
		}
		return state;
	}

	template <class Particles, std::size_t N_fields>
	void ExpectFieldsEqual(const Particles& actual, const Particles& expected,
		const std::array<std::vector<float> Particles::*, N_fields>& fields)
	{
		for (uint32_t field = 0; field < fields.size(); field++)
		{
			SCOPED_TRACE(field);
			const auto& actualValues = actual.*fields[field];
			const auto& expectedValues = expected.*fields[field];
			ASSERT_EQ(actualValues.size(), expectedValues.size());
			for (uint32_t i = 0; i < expectedValues.size(); i++)
			{
				SCOPED_TRACE(i);
				EXPECT_EQ(std::bit_cast<uint32_t>(actualValues[i]), std::bit_cast<uint32_t>(expectedValues[i]));
			}
		}
	}

	void ExpectStateEqual(const Sync::UniversalSimState& actual, const Sync::UniversalSimState& expected)
	{
		EXPECT_EQ(actual.step, expected.step);
		EXPECT_EQ(std::bit_cast<uint64_t>(actual.time), std::bit_cast<uint64_t>(expected.time));
		EXPECT_EQ(actual.particlesF.N_, expected.particlesF.N_);
		EXPECT_EQ(actual.particlesB.N_, expected.particlesB.N_);
		EXPECT_EQ(actual.particlesR.N_, expected.particlesR.N_);
		EXPECT_EQ(actual.rigidBodies.N_, expected.rigidBodies.N_);
		EXPECT_EQ(actual.particlesF.id, expected.particlesF.id);
		EXPECT_EQ(actual.particlesB.id, expected.particlesB.id);
		EXPECT_EQ(actual.particlesR.id, expected.particlesR.id);
		EXPECT_EQ(actual.particlesR.bodyId, expected.particlesR.bodyId);
		EXPECT_EQ(actual.rigidBodies.id, expected.rigidBodies.id);
		EXPECT_EQ(actual.rigidBodies.rigidParticleCount, expected.rigidBodies.rigidParticleCount);
		{ SCOPED_TRACE("fluid"); ExpectFieldsEqual(actual.particlesF, expected.particlesF, fluidFields); }
		{ SCOPED_TRACE("boundary"); ExpectFieldsEqual(actual.particlesB, expected.particlesB, boundaryFields); }
		{ SCOPED_TRACE("rigid"); ExpectFieldsEqual(actual.particlesR, expected.particlesR, rigidFields); }
		{ SCOPED_TRACE("bodies"); ExpectFieldsEqual(actual.rigidBodies, expected.rigidBodies, bodyFields); }
	}

	template <class Value>
	void AppendField(std::vector<uint8_t>& bytes, const std::vector<Value>& values)
	{
		for (const Value value : values)
		{
			uint64_t bits;
			if constexpr (std::is_same_v<Value, float>) { bits = std::bit_cast<uint32_t>(value); }
			else { bits = value; }
			for (uint32_t byte = 0; byte < sizeof(Value); byte++)
			{
				bytes.push_back(static_cast<uint8_t>((bits >> (8 * byte)) & 0xffu));
			}
		}
	}

	std::vector<uint8_t> ExpectedMixedStateBytes(const Sync::UniversalSimState& state)
	{
		// fixed mixed-state header, independent of the native header struct and production offset helpers
		std::vector<uint8_t> bytes{
			'T', 'S', 'P', 'S', 'I', 'M', '2', 0,
			4, 3, 2, 1, 64, 0, 0, 0,
			8, 7, 6, 5, 4, 3, 2, 1,
			0, 0, 0, 0, 0, 0, 0x29, 0x40,
			2, 0, 0, 0, 0, 0, 0, 0,
			1, 0, 0, 0, 0, 0, 0, 0,
			3, 0, 0, 0, 0, 0, 0, 0,
			2, 0, 0, 0, 0, 0, 0, 0 };
		AppendField(bytes, state.particlesF.id);
		for (const auto field : fluidFields) { AppendField(bytes, state.particlesF.*field); }
		AppendField(bytes, state.particlesB.id);
		for (const auto field : boundaryFields) { AppendField(bytes, state.particlesB.*field); }
		AppendField(bytes, state.particlesR.id);
		AppendField(bytes, state.particlesR.bodyId);
		for (const auto field : rigidFields) { AppendField(bytes, state.particlesR.*field); }
		AppendField(bytes, state.rigidBodies.id);
		AppendField(bytes, state.rigidBodies.rigidParticleCount);
		for (const auto field : bodyFields) { AppendField(bytes, state.rigidBodies.*field); }
		return bytes;
	}

	void SetUint64(std::vector<uint8_t>& bytes, const std::size_t offset, const uint64_t value)
	{
		for (uint32_t byte = 0; byte < 8; byte++)
		{
			bytes[offset + byte] = static_cast<uint8_t>((value >> (8 * byte)) & 0xffu);
		}
	}

	class StateSerializationTests : public testing::Test
	{
	protected:
		void SetUp() override
		{
			std::random_device random;
			const auto parent = std::filesystem::temp_directory_path();
			for (uint32_t attempt = 0; attempt < 16; attempt++)
			{
				const auto candidate = parent / ("tsp_state_tests_" + std::to_string(random()) + "_" + std::to_string(random()));
				if (!std::filesystem::create_directory(candidate)) { continue; }
				directory_ = candidate;
				Export::SaveOptions options;
				options.baseDir = directory_;
				serializer_ = std::make_unique<Export::SerializationMgr>(options);
				return;
			}
			FAIL() << "Could not create an isolated temporary directory";
		}

		void TearDown() override
		{
			if (directory_.empty()) { return; }
			std::error_code error;
			std::filesystem::remove_all(directory_, error);
			EXPECT_FALSE(error) << error.message();
		}

		std::vector<uint8_t> ReadBytes() const
		{
			std::ifstream input(directory_ / "state.bin", std::ios::binary);
			if (!input) { throw std::runtime_error("Could not open test state file"); }
			input.exceptions(std::ios::badbit);
			return std::vector<uint8_t>(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
		}

		void WriteBytes(const std::vector<uint8_t>& bytes) const
		{
			std::ofstream output(directory_ / "state.bin", std::ios::binary | std::ios::trunc);
			output.exceptions(std::ios::failbit | std::ios::badbit);
			if (!bytes.empty())
			{
				output.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
			}
			output.close();
		}

		std::filesystem::path directory_;
		std::unique_ptr<Export::SerializationMgr> serializer_;
	};

	// full restart state must retain every field, including signed zero and wide IDs
	TEST_F(StateSerializationTests, RoundTripsEveryStateField)
	{
		const auto expected = MakeState();
		Sync::UniversalSimState actual;
		serializer_->SaveUniversalSimState(expected, "state.bin");
		serializer_->LoadUniversalSimState(actual, "state.bin");

		ExpectStateEqual(actual, expected);
	}

	// explicit bytes catch field-order mistakes even when the writer and reader agree with each other
	TEST_F(StateSerializationTests, WritesExpectedBinaryLayout)
	{
		const auto state = MakeState();
		const auto expected = ExpectedMixedStateBytes(state);
		ASSERT_EQ(expected.size(), 604u);
		serializer_->SaveUniversalSimState(state, "state.bin");

		EXPECT_EQ(std::filesystem::file_size(directory_ / "state.bin"), 604u);
		EXPECT_EQ(ReadBytes(), expected);
	}

	// the loader must accept the documented layout without relying on the production writer
	TEST_F(StateSerializationTests, LoadsIndependentlyEncodedState)
	{
		const auto expected = MakeState();
		WriteBytes(ExpectedMixedStateBytes(expected));
		Sync::UniversalSimState actual;
		serializer_->LoadUniversalSimState(actual, "state.bin");

		ExpectStateEqual(actual, expected);
	}

	// missing populations contribute no arrays to the file or the loaded state
	TEST_F(StateSerializationTests, RoundTripsEmptyPopulations)
	{
		const std::array<std::array<uint32_t, 4>, 7> populations{{
			{ 0, 0, 0, 0 }, { 2, 0, 0, 0 }, { 0, 2, 0, 0 }, { 0, 0, 3, 2 },
			{ 0, 1, 3, 2 }, { 2, 0, 3, 2 }, { 2, 1, 0, 0 }
		}};
		for (uint32_t i = 0; i < populations.size(); i++)
		{
			SCOPED_TRACE(i);
			const auto& counts = populations[i];
			const auto expected = MakeState(counts[0], counts[1], counts[2], counts[3]);
			Sync::UniversalSimState actual;
			serializer_->SaveUniversalSimState(expected, "state.bin");
			serializer_->LoadUniversalSimState(actual, "state.bin");

			ExpectStateEqual(actual, expected);
			EXPECT_EQ(std::filesystem::file_size(directory_ / "state.bin"),
				64ull + 72ull * counts[0] + 28ull * counts[1] + 32ull * counts[2] + 136ull * counts[3]);
		}
	}

	// loading successively larger, smaller, and empty states must replace old arrays and metadata
	TEST_F(StateSerializationTests, ReplacesPreviouslyLoadedState)
	{
		std::array<Sync::UniversalSimState, 4> states{
			MakeState(5, 3, 4, 3), MakeState(1, 0, 1, 1), MakeState(0, 0, 0, 0), MakeState() };
		auto actual = MakeState(2, 2, 2, 2);
		for (uint32_t i = 0; i < states.size(); i++)
		{
			SCOPED_TRACE(i);
			auto& expected = states[i];
			expected.step += i;
			expected.time += 0.125 * i;
			serializer_->SaveUniversalSimState(expected, "state.bin");
			serializer_->LoadUniversalSimState(actual, "state.bin");

			ExpectStateEqual(actual, expected);
		}
	}

	// the header must be complete before its counts or metadata can be used
	TEST_F(StateSerializationTests, RejectsTruncatedHeaders)
	{
		const auto valid = ExpectedMixedStateBytes(MakeState());
		for (const std::size_t length : { 0u, 1u, 63u })
		{
			SCOPED_TRACE(length);
			WriteBytes(std::vector<uint8_t>(valid.begin(), valid.begin() + length));
			Sync::UniversalSimState actual;

			EXPECT_THROW(serializer_->LoadUniversalSimState(actual, "state.bin"), std::runtime_error);
		}
	}

	// magic, format version, byte order, and header size are checked independently of payload length
	TEST_F(StateSerializationTests, RejectsInvalidHeaderTags)
	{
		const auto valid = ExpectedMixedStateBytes(MakeState());
		for (const std::size_t offset : { 0u, 6u, 7u, 8u, 12u })
		{
			SCOPED_TRACE(offset);
			auto bytes = valid;
			bytes[offset] ^= 0xffu;
			WriteBytes(bytes);
			Sync::UniversalSimState actual;

			EXPECT_THROW(serializer_->LoadUniversalSimState(actual, "state.bin"), std::runtime_error);
		}
	}

	// header counts must describe exactly the arrays present in the file
	TEST_F(StateSerializationTests, RejectsMismatchedPopulationCounts)
	{
		const auto valid = ExpectedMixedStateBytes(MakeState());
		for (const std::size_t offset : { 32u, 40u, 48u, 56u })
		{
			SCOPED_TRACE(offset);
			auto bytes = valid;
			bytes[offset]++;
			WriteBytes(bytes);
			Sync::UniversalSimState actual;

			EXPECT_THROW(serializer_->LoadUniversalSimState(actual, "state.bin"), std::runtime_error);
		}
	}

	// both incomplete payloads and trailing bytes violate the exact file-size contract
	TEST_F(StateSerializationTests, RejectsIncorrectPayloadLengths)
	{
		const auto valid = ExpectedMixedStateBytes(MakeState());
		for (const std::size_t length : { 64u, 208u, 603u, 605u })
		{
			SCOPED_TRACE(length);
			auto bytes = valid;
			bytes.resize(length);
			WriteBytes(bytes);
			Sync::UniversalSimState actual;

			EXPECT_THROW(serializer_->LoadUniversalSimState(actual, "state.bin"), std::runtime_error);
		}
	}

	// invalid step and time values cannot be accepted as restart metadata
	TEST_F(StateSerializationTests, RejectsInvalidStepAndTime)
	{
		const auto valid = ExpectedMixedStateBytes(MakeState());
		for (const double time : { -1.0, std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN() })
		{
			auto bytes = valid;
			SetUint64(bytes, 24, std::bit_cast<uint64_t>(time));
			WriteBytes(bytes);
			Sync::UniversalSimState actual;

			EXPECT_THROW(serializer_->LoadUniversalSimState(actual, "state.bin"), std::runtime_error);
		}
		auto bytes = valid;
		SetUint64(bytes, 16, std::numeric_limits<uint64_t>::max());
		WriteBytes(bytes);
		Sync::UniversalSimState actual;
		EXPECT_THROW(serializer_->LoadUniversalSimState(actual, "state.bin"), std::runtime_error);
	}

	// a valid file size does not make the invalid particle ID sentinel acceptable
	TEST_F(StateSerializationTests, RejectsInvalidParticleIds)
	{
		const auto valid = ExpectedMixedStateBytes(MakeState());
		for (const std::size_t offset : { 64u, 208u, 236u })
		{
			SCOPED_TRACE(offset);
			auto bytes = valid;
			SetUint64(bytes, offset, std::numeric_limits<uint64_t>::max());
			WriteBytes(bytes);
			Sync::UniversalSimState actual;

			EXPECT_THROW(serializer_->LoadUniversalSimState(actual, "state.bin"), std::runtime_error);
		}
	}

	// invalid IDs in any population must be rejected before publishing a restart file
	TEST_F(StateSerializationTests, RejectsSavingInvalidParticleIds)
	{
		for (uint32_t population = 0; population < 3; population++)
		{
			SCOPED_TRACE(population);
			auto state = MakeState();
			const std::array ids{ &state.particlesF.id, &state.particlesB.id, &state.particlesR.id };
			(*ids[population])[0] = std::numeric_limits<uint64_t>::max();

			EXPECT_THROW(serializer_->SaveUniversalSimState(state, "state.bin"), std::runtime_error);
			EXPECT_FALSE(std::filesystem::exists(directory_ / "state.bin"));
		}
	}

	// rigid particles cannot be saved with references outside the body array
	TEST_F(StateSerializationTests, RejectsSavingOutOfRangeRigidBodyIndices)
	{
		auto state = MakeState();
		state.particlesR.bodyId[0] = state.rigidBodies.N_;

		EXPECT_THROW(serializer_->SaveUniversalSimState(state, "state.bin"), std::runtime_error);
		EXPECT_FALSE(std::filesystem::exists(directory_ / "state.bin"));
	}

	// saved body counts must agree with the rigid particle memberships
	TEST_F(StateSerializationTests, RejectsSavingMismatchedRigidBodyParticleCounts)
	{
		auto state = MakeState();
		state.rigidBodies.rigidParticleCount[0]--;

		EXPECT_THROW(serializer_->SaveUniversalSimState(state, "state.bin"), std::runtime_error);
		EXPECT_FALSE(std::filesystem::exists(directory_ / "state.bin"));
	}

	// rigid particles must reference existing bodies and agree with the saved per-body counts
	TEST_F(StateSerializationTests, RejectsInvalidRigidAttachments)
	{
		const auto valid = ExpectedMixedStateBytes(MakeState());
		for (const std::size_t offset : { 260u, 340u })
		{
			SCOPED_TRACE(offset);
			auto bytes = valid;
			bytes[offset] = 2;
			if (offset == 340) { bytes[offset] = 1; }
			WriteBytes(bytes);
			Sync::UniversalSimState actual;

			EXPECT_THROW(serializer_->LoadUniversalSimState(actual, "state.bin"), std::runtime_error);
		}
	}

	// population offsets follow the wire widths without allocating any particle arrays
	TEST(SimStateFormatTests, ComputesMixedPopulationOffsets)
	{
		const auto header = Export::MakeSimStateHeader(7, 0.5, 2, 1, 3, 2);
		const auto actual = Export::ComputeSimStateOffsets(header);

		EXPECT_EQ(actual.fluid, 64u);
		EXPECT_EQ(actual.boundary, 208u);
		EXPECT_EQ(actual.rigid, 236u);
		EXPECT_EQ(actual.rigidBodies, 332u);
		EXPECT_EQ(actual.totalBytes, 604u);
	}

	// absent populations leave every offset at the end of the header
	TEST(SimStateFormatTests, EmptyPopulationsContainOnlyHeader)
	{
		const auto header = Export::MakeSimStateHeader(0, 0.0, 0, 0, 0, 0);
		const auto actual = Export::ComputeSimStateOffsets(header);

		EXPECT_EQ(actual.fluid, 64u);
		EXPECT_EQ(actual.boundary, 64u);
		EXPECT_EQ(actual.rigid, 64u);
		EXPECT_EQ(actual.rigidBodies, 64u);
		EXPECT_EQ(actual.totalBytes, 64u);
	}

	// byte offsets remain wide enough for states larger than four GiB
	TEST(SimStateFormatTests, KeepsOffsetsWiderThan32Bits)
	{
		const auto header = Export::MakeSimStateHeader(0, 0.0, 67'108'864, 0, 0, 0);
		const auto actual = Export::ComputeSimStateOffsets(header);

		EXPECT_EQ(actual.fluid, 64u);
		EXPECT_EQ(actual.boundary, 4'831'838'272ull);
		EXPECT_EQ(actual.rigid, 4'831'838'272ull);
		EXPECT_EQ(actual.rigidBodies, 4'831'838'272ull);
		EXPECT_EQ(actual.totalBytes, 4'831'838'272ull);
	}

	// field-size multiplication and combined header-plus-payload addition must not wrap
	TEST(SimStateFormatTests, RejectsOverflowingOffsets)
	{
		constexpr std::array counts{
			&Export::BinHeaderSim::N_F, &Export::BinHeaderSim::N_B,
			&Export::BinHeaderSim::N_R, &Export::BinHeaderSim::N_RB };
		for (const auto count : counts)
		{
			auto header = Export::MakeSimStateHeader(0, 0.0, 0, 0, 0, 0);
			header.*count = std::numeric_limits<uint64_t>::max();

			EXPECT_THROW(Export::ComputeSimStateOffsets(header), std::runtime_error);
		}
		const auto header = Export::MakeSimStateHeader(0, 0.0,
			std::numeric_limits<uint64_t>::max() / 72, 0, 0, 0);
		EXPECT_THROW(Export::ComputeSimStateOffsets(header), std::runtime_error);
	}
}
