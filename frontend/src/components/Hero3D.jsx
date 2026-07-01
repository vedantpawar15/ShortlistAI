import React, { useRef } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { Float, ContactShadows, Environment, MeshDistortMaterial, Sphere } from '@react-three/drei';

function AnimatedShapes() {
  const sphereRef = useRef();

  useFrame((state) => {
    if (sphereRef.current) {
      sphereRef.current.rotation.x = state.clock.getElapsedTime() * 0.2;
      sphereRef.current.rotation.y = state.clock.getElapsedTime() * 0.3;
    }
  });

  return (
    <>
      <Float speed={2} rotationIntensity={1} floatIntensity={2}>
        <Sphere ref={sphereRef} args={[1, 64, 64]} scale={1.5} position={[0, 0, 0]}>
          <MeshDistortMaterial
            color="#2563eb"
            attach="material"
            distort={0.4}
            speed={2}
            roughness={0.2}
            metalness={0.8}
          />
        </Sphere>
      </Float>

      <Float speed={3} rotationIntensity={2} floatIntensity={3}>
        <mesh position={[2, 1, -2]} scale={0.5} rotation={[Math.PI / 4, Math.PI / 4, 0]}>
          <boxGeometry args={[1, 1, 1]} />
          <meshStandardMaterial color="#10b981" roughness={0.1} metalness={0.6} />
        </mesh>
      </Float>

      <Float speed={1.5} rotationIntensity={1.5} floatIntensity={1.5}>
        <mesh position={[-2, -1.5, -1]} scale={0.6}>
          <torusGeometry args={[0.8, 0.25, 16, 32]} />
          <meshStandardMaterial color="#8b5cf6" roughness={0.1} metalness={0.5} />
        </mesh>
      </Float>
    </>
  );
}

export default function Hero3D() {
  return (
    <div style={{ width: '150%', height: '600px', position: 'absolute', right: '-25%', top: '50%', transform: 'translateY(-50%)', zIndex: 0 }}>
      <Canvas camera={{ position: [0, 0, 5], fov: 45 }}>
        <ambientLight intensity={0.5} />
        <directionalLight position={[10, 10, 5]} intensity={1} />
        <directionalLight position={[-10, -10, -5]} intensity={0.5} />
        <Environment preset="city" />
        <AnimatedShapes />
        <ContactShadows position={[0, -2.5, 0]} opacity={0.4} scale={10} blur={2} far={4} />
      </Canvas>
    </div>
  );
}
