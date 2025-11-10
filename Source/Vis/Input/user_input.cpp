// user_input.cpp

// external
#include <GLFW/glfw3.h>
#include <spdlog/spdlog.h>

// internal
#include "user_input.h"



using namespace Input;

UserInputMgr::UserInputMgr(Sim::SimParamsMgr& simParamsMgr,
						   Vis::VisParamsMgr& visParamsMgr)
	: simParamsMgr_{ simParamsMgr }
	, visParamsMgr_{ visParamsMgr }
{
	// handled by Init()
}

void UserInputMgr::Init(const Vis::VisParams& visParams)
{
	// adopt initial camera from params snapshot
	defaultCamera_ = visParams.camera;
}

void UserInputMgr::Shutdown()
{
	// TODO: implement
}

// rebuilds camera matrices and pushes to the parameter manager
void UserInputMgr::UpdateCamera(const Vis::CameraState& camera,
								const Vis::FrameBufferConfig& frameBuffer, float dt)
{
	float tempYawDeg = camera.yawDeg;
	float tempPitchDeg = camera.pitchDeg;
	if (keys_[GLFW_KEY_Q]) // turn left
	{
		tempYawDeg -= camera.turnSpeedDeg * dt;
	}
	if (keys_[GLFW_KEY_E]) // turn right
	{
		tempYawDeg += camera.turnSpeedDeg * dt;
	}
	if (camera.yawDeg > 360.0f ||
		camera.yawDeg < -360.0f) // keep it in a reasonable range
	{
		tempYawDeg   = std::fmod(camera.yawDeg, 360.0f);
		tempPitchDeg = glm::clamp(camera.pitchDeg, -89.0f, 89.0f);
	}

	// build camera basis from yaw/pitch
	const float yaw	  = glm::radians(camera.yawDeg);
	const float pitch = glm::radians(camera.pitchDeg);

	glm::vec3 forward
	{
		std::cos(pitch) * std::cos(yaw),
		std::sin(pitch),
		std::cos(pitch) * std::sin(yaw)
	};
	forward = glm::normalize(forward);

	constexpr glm::vec3 worldUp{ 0.0f, 1.0f, 0.0f };
	glm::vec3 right = glm::normalize(glm::cross(forward, worldUp));
	glm::vec3 up    = glm::normalize(glm::cross(right, forward));

	// compute movement speed with modifiers
	float tempMoveSpeed = camera.moveSpeed;
	if (keys_[GLFW_KEY_LEFT_SHIFT] || keys_[GLFW_KEY_RIGHT_SHIFT])
	{
		// tempMoveSpeed *= camera.speedMult;
	}

	// integrate position
	glm::vec3 delta{ 0.0f };
	if (keys_[GLFW_KEY_W])				{ delta += forward; }	// forward
	if (keys_[GLFW_KEY_S])				{ delta -= forward; }   // backward
	if (keys_[GLFW_KEY_D])				{ delta += right; }     // right
	if (keys_[GLFW_KEY_A])				{ delta -= right; }     // left
	if (keys_[GLFW_KEY_UP])				{ delta += worldUp; }   // up
	if (keys_[GLFW_KEY_DOWN])			{ delta -= worldUp; }   // down

	glm::vec3 tempCamPos = camera.camPos;
	if (glm::dot(delta, delta) > 0.000001f)
	{
		tempCamPos += glm::normalize(delta) * tempMoveSpeed * dt;
	}

	// build view matrix
	const glm::mat4 tempViewMat = glm::lookAt(tempCamPos, tempCamPos + forward, up);

	// build projection matrix
	float aspect = 1.0f;
	if (frameBuffer.size.y > 0)
	{
		aspect = static_cast<float>(frameBuffer.size.x)
			   / static_cast<float>(frameBuffer.size.y);
	}
	const glm::mat4 temoProjMat = glm::perspective(glm::radians(camera.yFovDeg), aspect,
												   camera.zNear, camera.zFar);

	// push to visParamsMgr
	visParamsMgr_.SetParam(Vis::CameraPatch
		{
			.viewMat = tempViewMat,
			.projMat = temoProjMat,
			.camPos = tempCamPos,
			.yawDeg = tempYawDeg,
			.pitchDeg = tempPitchDeg
		});
}

bool UserInputMgr::OnKey(const Sim::SimParams& simParams, const Vis::CameraState& camera, int key, int scancode, int action, int mods)
{
	// ignore unknown keys
	if (key < 0 || key >= static_cast<int>(keys_.size())) { return false; }

	// track pressed state for continuous movement
	if (action == GLFW_PRESS)			{ keys_.set(key, true); }
	else if (action == GLFW_RELEASE)	{ keys_.set(key, false); }

	// one-shot actions on press only
	if (action == GLFW_PRESS)
	{
		switch (key)
		{
		case GLFW_KEY_SPACE: // toggle simulation pause
		{
			bool toggle = !simParams.simRunning;
			simParamsMgr_.Set_SimRunning(toggle);

			return true;
		}
		case GLFW_KEY_RIGHT: // unpause sim for one step
		{
			simParamsMgr_.Set_SimRunNextStep(true);

			return true;
		}
		case GLFW_KEY_R: // reset camera to default
		{
			visParamsMgr_.SetParam(Vis::CameraPatch
				{
					.viewMat = defaultCamera_.viewMat,
					.projMat = defaultCamera_.projMat,
					.camPos = defaultCamera_.camPos,
					.yawDeg = defaultCamera_.yawDeg,
					.pitchDeg = defaultCamera_.pitchDeg,
					.moveSpeed = defaultCamera_.moveSpeed,
					.turnSpeedDeg = defaultCamera_.turnSpeedDeg,
					.yFovDeg = defaultCamera_.yFovDeg,
					.zNear = defaultCamera_.zNear,
					.zFar = defaultCamera_.zFar
				});

			return true;
		}

		case GLFW_KEY_F: // place camera at a fixed radius looking at origin
		{
			const float yaw   = glm::radians(camera.yawDeg);
			const float pitch = glm::radians(camera.pitchDeg);

			glm::vec3 forward
			{
				std::cos(pitch) * std::cos(yaw),
				std::sin(pitch),
				std::cos(pitch) * std::sin(yaw)
			};
			forward = glm::normalize(forward);

			// distance hardcoded as 5.0f for now
			glm::vec3 tempCamPos = - forward * 5.0f;

			visParamsMgr_.SetParam(Vis::CameraPatch
				{
					.camPos = tempCamPos,
				});

			return true;
		}

		case GLFW_KEY_X: // toggle base movement speed
		{
			float tempMoveSpeed = camera.moveSpeed
								< (defaultCamera_.moveSpeed + 0.000001f)
								? defaultCamera_.moveSpeed * camera.speedMult
								: defaultCamera_.moveSpeed;

			visParamsMgr_.SetParam(Vis::CameraPatch
				{
					.moveSpeed = tempMoveSpeed,
				});

			return true;
		}

		default:
			break;
		}
	}

	(void)mods;

	return true;
}

bool UserInputMgr::OnMouseButton(int button, int action, int mods)
{
	// TODO: add functionality

	return false;
}

bool UserInputMgr::OnCursorPos(double xPos, double yPos)
{
	// TODO: add functionality

	return true;
}

bool UserInputMgr::OnScroll(double xOff, double yOff)
{
	// TODO: add functionality

	return true;
}
