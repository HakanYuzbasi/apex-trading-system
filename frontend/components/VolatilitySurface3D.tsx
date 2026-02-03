"use client";

import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, PerspectiveCamera, Grid } from "@react-three/drei";
import { useMemo, useRef } from "react";
import * as THREE from "three";

function SurfaceMesh() {
    const meshRef = useRef<THREE.Mesh>(null);

    // Generate dummy vol surface data
    const { geometry, colors } = useMemo(() => {
        const size = 30;
        const geometry = new THREE.PlaneGeometry(10, 10, size, size);
        const count = geometry.attributes.position.count;

        // Deform plane based on "Vol Smile" logic
        // z = f(x, y) where x=strike, y=expiry
        const positions = geometry.attributes.position;

        // Create color buffer
        const colors = new Float32Array(count * 3);

        for (let i = 0; i < count; i++) {
            const x = positions.getX(i); // Strike
            const y = positions.getY(i); // Expiry

            // Volatility Function
            // Skew (Smile): increases as abs(x) increases
            const skew = Math.pow(x * 0.3, 2);
            // Term structure: slight increase with y
            const term = y * 0.1;

            const z = skew + term + 0.5;

            positions.setZ(i, z);

            // Color map based on Z height (Vol)
            // Low vol (blue/cyan) -> High vol (purple/red)
            const t = Math.min(1, Math.max(0, (z - 0.5) / 2));

            // Simple Gradient: Cyan (0,1,1) -> Red (1,0,0)
            // R: t, G: 1-t, B: 1-t*0.5
            colors[i * 3] = t;     // R
            colors[i * 3 + 1] = 1 - t; // G
            colors[i * 3 + 2] = 1 - t; // B
        }

        geometry.computeVertexNormals();
        geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

        return { geometry, colors };
    }, []);

    useFrame((state) => {
        if (meshRef.current) {
            // Slowly rotate
            meshRef.current.rotation.z += 0.001;
        }
    });

    return (
        <mesh ref={meshRef} geometry={geometry} rotation={[-Math.PI / 2.5, 0, 0]}>
            {/* Wireframe overlay for "Tech" look */}
            <meshStandardMaterial
                vertexColors
                roughness={0.2}
                metalness={0.8}
                side={THREE.DoubleSide}
                wireframe={false}
            />
        </mesh>
    );
}

function WireframeOverlay() {
    // Duplicate of surface but wireframe
    return (
        <mesh position={[0, 0, 0.01]} rotation={[-Math.PI / 2.5, 0, 0]}>
            <planeGeometry args={[10, 10, 30, 30]} />
            {/* We need the same deformation, so maybe better to pass geometry props. 
                 For simplicity, skip dynamic overlay or use helper provided by Drei if needed.
             */}
        </mesh>
    )
}

export default function VolatilitySurface3D() {
    return (
        <div style={{ width: '100%', height: '100%', minHeight: 400, position: 'relative' }}>
            <Canvas style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%' }}>
                <PerspectiveCamera makeDefault position={[0, -10, 8]} fov={50} />
                <OrbitControls enableZoom={false} autoRotate autoRotateSpeed={0.5} />

                <ambientLight intensity={0.5} />
                <pointLight position={[10, 10, 10]} intensity={1} color="#00f3ff" />
                <pointLight position={[-10, -10, 10]} intensity={1} color="#7000ff" />

                <SurfaceMesh />

                <Grid
                    args={[20, 20]}
                    cellColor="#333"
                    sectionColor="#00f3ff"
                    fadeDistance={20}
                    fadeStrength={1}
                    position={[0, 0, -1]}
                />
            </Canvas>
        </div>
    );
}
