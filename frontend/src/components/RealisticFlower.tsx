import React from 'react';
import styled, { keyframes } from 'styled-components';
import { FlowerType } from '../types';

interface RealisticFlowerProps {
  x: number;
  y: number;
  size: number;
  flowerType: FlowerType;
  rotation: number;
  opacity: number;
  bloomProgress: number;
}

const gentleFloat = keyframes`
  0%, 100% { 
    transform: translateY(0) translateX(0) rotate(0deg) scale(1);
  }
  25% { 
    transform: translateY(-3px) translateX(2px) rotate(1deg) scale(1.02);
  }
  50% { 
    transform: translateY(-1px) translateX(-2px) rotate(-1deg) scale(0.98);
  }
  75% { 
    transform: translateY(-4px) translateX(1px) rotate(0.5deg) scale(1.01);
  }
`;

const bloomIn = keyframes`
  0% { 
    transform: scale(0) rotate(-180deg);
    opacity: 0;
  }
  50% { 
    transform: scale(1.1) rotate(0deg);
    opacity: 0.8;
  }
  100% { 
    transform: scale(1) rotate(0deg);
    opacity: 1;
  }
`;

const petalShimmer = keyframes`
  0%, 100% { filter: brightness(1) saturate(1); }
  50% { filter: brightness(1.1) saturate(1.2); }
`;

const FlowerContainer = styled.div<{
  $x: number;
  $y: number;
  $size: number;
  $rotation: number;
  $opacity: number;
}>`
  position: fixed;
  left: ${props => props.$x}px;
  top: ${props => props.$y}px;
  width: ${props => props.$size}px;
  height: ${props => props.$size}px;
  transform: rotate(${props => props.$rotation}deg);
  opacity: ${props => props.$opacity};
  pointer-events: none;
  z-index: 1;
  animation: ${gentleFloat} 12s ease-in-out infinite;
  filter: drop-shadow(0 4px 12px rgba(0, 0, 0, 0.15));
  will-change: transform, opacity;
`;

const FlowerBase = styled.div`
  position: relative;
  width: 100%;
  height: 100%;
  animation: ${bloomIn} 3s cubic-bezier(0.34, 1.56, 0.64, 1);
`;

const Petal = styled.div<{
  $color: string;
  $index: number;
  $totalPetals: number;
}>`
  position: absolute;
  top: 50%;
  left: 50%;
  width: 45%;
  height: 25%;
  background: linear-gradient(
    45deg,
    ${props => props.$color}f0,
    ${props => props.$color}e0,
    ${props => props.$color}ff,
    ${props => props.$color}d0
  );
  border-radius: 80% 20% 80% 20%;
  transform-origin: 15% 90%;
  transform: translate(-15%, -90%) rotate(${props => (360 / props.$totalPetals) * props.$index}deg);
  box-shadow: 
    inset 0 1px 3px rgba(255, 255, 255, 0.4),
    inset 0 -1px 2px rgba(0, 0, 0, 0.1),
    0 2px 6px rgba(0, 0, 0, 0.15);
  animation: ${petalShimmer} 8s ease-in-out infinite;
  animation-delay: ${props => props.$index * 0.2}s;
  
  &::before {
    content: '';
    position: absolute;
    top: 15%;
    left: 20%;
    width: 60%;
    height: 70%;
    background: linear-gradient(
      to bottom right,
      rgba(255, 255, 255, 0.5),
      rgba(255, 255, 255, 0.2),
      transparent
    );
    border-radius: 60% 40% 60% 40%;
    transform: rotate(-10deg);
  }
  
  &::after {
    content: '';
    position: absolute;
    top: 5%;
    left: 5%;
    width: 20%;
    height: 15%;
    background: rgba(255, 255, 255, 0.7);
    border-radius: 50%;
    filter: blur(1px);
  }
`;

const FlowerCenter = styled.div<{ $color: string; $size: number }>`
  position: absolute;
  top: 50%;
  left: 50%;
  width: ${props => props.$size * 0.25}px;
  height: ${props => props.$size * 0.25}px;
  background: radial-gradient(
    circle at 30% 30%,
    ${props => props.$color}ff,
    ${props => props.$color}dd,
    ${props => props.$color}bb
  );
  border-radius: 50%;
  transform: translate(-50%, -50%);
  box-shadow: 
    inset 0 2px 4px rgba(255, 255, 255, 0.3),
    inset 0 -2px 4px rgba(0, 0, 0, 0.2),
    0 2px 8px rgba(0, 0, 0, 0.2);
    
  &::before {
    content: '';
    position: absolute;
    top: 25%;
    left: 25%;
    width: 50%;
    height: 50%;
    background: radial-gradient(
      circle at 40% 40%,
      rgba(255, 255, 255, 0.8),
      rgba(255, 255, 255, 0.3),
      transparent
    );
    border-radius: 50%;
  }
`;

const RealisticFlower: React.FC<RealisticFlowerProps> = React.memo(({
  x,
  y,
  size,
  flowerType,
  rotation,
  opacity,
  bloomProgress
}) => {
  // 使用useMemo缓存花瓣渲染
  const petals = React.useMemo(() => {
    const petalElements = [];
    for (let i = 0; i < flowerType.petals; i++) {
      const colorIndex = i % flowerType.colors.length;
      petalElements.push(
        <Petal
          key={i}
          $color={flowerType.colors[colorIndex]}
          $index={i}
          $totalPetals={flowerType.petals}
        />
      );
    }
    return petalElements;
  }, [flowerType.petals, flowerType.colors]);

  return (
    <FlowerContainer
      $x={x}
      $y={y}
      $size={size}
      $rotation={rotation}
      $opacity={opacity * bloomProgress}
    >
      <FlowerBase>
        {petals}
        <FlowerCenter $color={flowerType.center} $size={size} />
      </FlowerBase>
    </FlowerContainer>
  );
}, (prevProps, nextProps) => {
  // 更宽松的比较函数，减少微小变化导致的重新渲染
  const positionThreshold = 1.0; // 增大位置变化阈值
  const rotationThreshold = 2.0; // 增大旋转变化阈值
  const opacityThreshold = 0.02; // 增大透明度变化阈值
  
  return (
    Math.abs(prevProps.x - nextProps.x) < positionThreshold &&
    Math.abs(prevProps.y - nextProps.y) < positionThreshold &&
    Math.abs(prevProps.rotation - nextProps.rotation) < rotationThreshold &&
    Math.abs(prevProps.opacity - nextProps.opacity) < opacityThreshold &&
    prevProps.size === nextProps.size &&
    prevProps.flowerType === nextProps.flowerType &&
    prevProps.bloomProgress === nextProps.bloomProgress
  );
});

RealisticFlower.displayName = 'RealisticFlower';

export default RealisticFlower;
