// shader.h
//
// TODO: comment

#pragma once



namespace Render
{
	static const char* VS_impostor = R"(#version 450 core

	// per-vertex attributes for a quad (2 triangles / 4 vertices) centered at origin
	layout(location = 0) in vec2 aCorner;		// per-vertex: (-1,-1), (1,-1), (1,1), (-1,1)
	layout(location = 1) in vec4 aPosAndRad;	// per-instance: (x, y, z, radius)
	layout(location = 2) in float aScalar;		// per-instance: scalar value for color coding

	layout(std140, binding = 1) uniform Camera
	{
		mat4 uViewMat;
		mat4 uProjMat;
		mat4 uViewProjMat;
		vec4 uCamPos;							// xyz = camera position in world space, w unused (0)
	};

	// output passed to fragment shader for round mask
	noperspective out vec2 vCorner;
	out vec3 vCenterVS;							// sphere center in view space
	out float vRadius;							// sphere radius in view space units
	out float vScalar;							// scalar value for color coding
	out vec3 vBillboardVS;						// position of billboard vertex in view space

	void main()
	{
		// transform sphere center from world to view space
		vec3 centerVS = (uViewMat * vec4(aPosAndRad.xyz, 1.0)).xyz;

		// build a camera-facing quad in view space by offsetting along right and up vectors
		// (makes quad large enough to cover the sphere, fragment shader discards pixels outside)
		vec3 rightVS = vec3(1.0, 0.0, 0.0);
		vec3 upVS = vec3(0.0, 1.0, 0.0);
		vec3 billboardVS = centerVS + (aCorner.x * rightVS + aCorner.y * upVS) * aPosAndRad.w;

		// final clip space position
		gl_Position = uProjMat * vec4(billboardVS, 1.0);

		// pass variables to fragment shader
		vCorner = aCorner;
		vCenterVS = centerVS;
		vRadius = aPosAndRad.w;
		vScalar = aScalar;
		vBillboardVS = billboardVS;
	}
	)";

	static const char* FS_impostor = R"(#version 450 core

	// input from vertex shader
	noperspective in vec2 vCorner;
	in vec3 vCenterVS;							
	in float vRadius;
	in float vScalar;
	in vec3 vBillboardVS;

	// declare explicit color output
	layout(location = 0) out vec4 FragColor;
	
	// explicitly enable early depth testing
	layout(depth_less) out float gl_FragDepth;

	layout(std140, binding = 0) uniform Params
	{
		vec4 uLightDirVS;						// xyz = light direction in view space, w unused (0)
		vec4 uBaseColorAlpha;					// xyz = base color, w = alpha
		vec4 uScalarMinMaxUse;					// x = min, y = max, z = useScalar (0/1), w unused (0)
	};

    layout(std140, binding = 1) uniform Camera
	{
		mat4 uViewMat;
		mat4 uProjMat;
		mat4 uViewProjMat;
		vec4 uCamPos;							// xyz = camera position in world space, w unused (0)
	};

	// bind color lookup-table (256 entries) to texture unit 0
	layout(binding = 0) uniform sampler1D uColorMap;

	void main()
	{
		// skip pixels outside unit circle of the quad (avoids ray-sphere tests)
		if (dot(vCorner, vCorner) > 1.000001) { discard; }

		// reconstruct ray direction in view space
		vec3 d = normalize(vBillboardVS);

		// sphere parameters in view space
		vec3 C = vCenterVS;
		float R = vRadius;

		// ray-sphere intersection
		float b = dot(d, -C);
		float c = dot(C, C) - R * R;
		float disc = b * b - c;

		if (disc < 0.0) { discard; }			// no intersection

		float t = - b - sqrt(disc);				// nearest intersection
		if (t < 0.0) { discard; }				// intersection behind camera

		// hit point and normal in view space
		vec3 P = t * d;
		vec3 N = normalize(P - C);

		// compute correct depth from the hit point P
		vec4 clipPos = uProjMat * vec4(P, 1.0);
		float ndcDepth = clipPos.z / clipPos.w;
		gl_FragDepth = ndcDepth * 0.5 + 0.5;	// map from [-1, 1] to [0, 1]

		// color selection
		vec3 lightVS = normalize(uLightDirVS.xyz);
		vec3 baseColor = uBaseColorAlpha.xyz;
		float alpha = uBaseColorAlpha.w;
		float scalarMin = uScalarMinMaxUse.x;
		float scalarMax = uScalarMinMaxUse.y;
		bool useScalar = (uScalarMinMaxUse.z > 0.5);

		// normalize scalar to [0, 1] with clamping
		float u = clamp((vScalar - scalarMin) / max(scalarMax - scalarMin, 0.000001), 0.0, 1.0);

		// choose RGB color
		vec3 color = useScalar ? texture(uColorMap, u).xyz
							   : baseColor;

		// simple Blinn-Phong shading in view space
		vec3 viewDir = normalize(-P);
		vec3 lightDir = lightVS;
		vec3 halfVec = normalize(lightDir + viewDir);

		// diffuse and specular lighting (shininess hardcoded to 16.0)
		float diff = max(dot(N, lightDir), 0.0);
		float spec = pow(max(dot(N, halfVec), 0.0), 16.0);

		vec3 ambient = 0.1 * color;
		vec3 lit = ambient + diff * color + spec * vec3(1.0);

		// output final color
		FragColor = vec4(lit, alpha);
	}
	)";

} // namespace Render
