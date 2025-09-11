import React, { useEffect, useRef, useCallback } from 'react';
import { Flower } from '../types';
import { FLOWER_TYPES } from '../constants';
import RealisticFlower from './RealisticFlower';

const FloatingFlowers: React.FC = () => {
  const flowersRef = useRef<Flower[]>([]);
  const animationRef = useRef<number>();
  const lastTimeRef = useRef<number>(0);
  const containerRef = useRef<HTMLDivElement>(null);

  const createFlower = useCallback((): Flower => {
    const speed = Math.random() * 30 + 20; // 20-50 像素/秒的速度
    const angle = Math.random() * Math.PI * 2;
    
    return {
      id: Math.random().toString(36).substr(2, 9),
      x: Math.random() * window.innerWidth,
      y: Math.random() * window.innerHeight,
      size: Math.random() * 25 + 15, // 15-40px 尺寸范围
      flowerType: FLOWER_TYPES[Math.floor(Math.random() * FLOWER_TYPES.length)],
      speed: speed,
      angle: angle,
      rotation: Math.random() * 360,
      opacity: 0.65, // 固定初始透明度
      bloomProgress: Math.random() * 0.2 + 0.8, // 0.8-1.0 绽放程度
      // 初始化速度向量，与角度保持一致
      vx: Math.cos(angle) * speed,
      vy: Math.sin(angle) * speed,
      targetAngle: angle, // 初始目标角度与当前角度一致
      wanderTimer: Math.random() * 150, // 随机初始计时器，避免同时改变方向
      maxWanderTime: Math.random() * 200 + 150, // 1.5-3.5秒改变一次方向
    };
  }, []);

  // 更自然且平滑的运动算法
  const updateFlower = useCallback((flower: Flower, deltaTime: number) => {
    // 标准化时间步长，限制最大值防止大的跳跃
    const dt = Math.min(deltaTime / 1000, 0.032); // 最大32ms，防止卡顿时大幅跳跃

    // 游走行为：定期改变目标方向
    flower.wanderTimer += deltaTime;
    if (flower.wanderTimer >= flower.maxWanderTime) {
      flower.targetAngle = Math.random() * Math.PI * 2;
      flower.wanderTimer = 0;
      flower.maxWanderTime = Math.random() * 200 + 150; // 缩短间隔，让运动更自然
    }

    // 更平滑的转向算法
    const angleDiff = flower.targetAngle - flower.angle;
    const normalizedAngleDiff = Math.atan2(Math.sin(angleDiff), Math.cos(angleDiff));
    
    // 减少转向速度，让转向更自然
    const turnSpeed = 0.8; // 降低转向速度
    flower.angle += normalizedAngleDiff * turnSpeed * dt;

    // 减少随机噪音的频率和强度
    if (Math.random() < 0.1) { // 只有10%的概率添加噪音
      const noise = (Math.random() - 0.5) * 0.05;
      flower.angle += noise * dt;
    }

    // 使用更平滑的速度插值而不是直接计算
    const targetVx = Math.cos(flower.angle) * flower.speed;
    const targetVy = Math.sin(flower.angle) * flower.speed;
    
    // 平滑插值到目标速度
    const smoothing = 0.95; // 平滑因子
    flower.vx = flower.vx * smoothing + targetVx * (1 - smoothing);
    flower.vy = flower.vy * smoothing + targetVy * (1 - smoothing);

    // 更新位置
    flower.x += flower.vx * dt;
    flower.y += flower.vy * dt;

    // 柔和的边界处理
    const margin = 50;
    if (flower.x < -margin) {
      flower.x = window.innerWidth + margin;
    } else if (flower.x > window.innerWidth + margin) {
      flower.x = -margin;
    }

    if (flower.y < -margin) {
      flower.y = window.innerHeight + margin;
    } else if (flower.y > window.innerHeight + margin) {
      flower.y = -margin;
    }

    // 更平滑的旋转
    flower.rotation += 20 * dt; // 固定旋转速度
    
    // 更稳定的透明度变化
    const time = Date.now() * 0.0005; // 减慢透明度变化速度
    const breathingOpacity = Math.sin(time + flower.x * 0.005) * 0.05; // 减小变化幅度
    flower.opacity = Math.max(0.5, Math.min(0.8, 
      0.65 + breathingOpacity // 固定基础透明度
    ));

    return flower;
  }, []);

  // 响应式花朵数量计算
  const calculateFlowerCount = useCallback(() => {
    const area = window.innerWidth * window.innerHeight;
    const density = 20000; // 每20000像素一个花朵
    return Math.max(8, Math.min(20, Math.floor(area / density)));
  }, []);

  // 初始化和窗口大小变化处理
  const handleResize = useCallback(() => {
    const newCount = calculateFlowerCount();
    const currentCount = flowersRef.current.length;
    
    if (newCount > currentCount) {
      // 添加花朵
      for (let i = currentCount; i < newCount; i++) {
        flowersRef.current.push(createFlower());
      }
    } else if (newCount < currentCount) {
      // 移除花朵
      flowersRef.current = flowersRef.current.slice(0, newCount);
    }
  }, [calculateFlowerCount, createFlower]);

  // 使用状态强制组件重新渲染
  const [, forceUpdate] = React.useReducer(x => x + 1, 0);

  // 节流渲染更新
  const lastRenderTime = useRef<number>(0);
  const RENDER_THROTTLE = 1000 / 24; // 24fps 渲染频率，更稳定的帧率



  // 可见性检测，优化性能
  const isVisible = useRef<boolean>(true);

  // 优化动画函数，添加可见性检查
  const optimizedAnimateWithRender = useCallback((currentTime: number) => {
    if (!isVisible.current) return;

    const deltaTime = currentTime - lastTimeRef.current;
    lastTimeRef.current = currentTime;

    // 更新所有花朵
    flowersRef.current.forEach(flower => updateFlower(flower, deltaTime));

    // 节流重新渲染，避免每帧都更新
    if (currentTime - lastRenderTime.current >= RENDER_THROTTLE) {
      forceUpdate();
      lastRenderTime.current = currentTime;
    }

    animationRef.current = requestAnimationFrame(optimizedAnimateWithRender);
  }, [updateFlower]);

  // 主要useEffect，处理初始化和清理
  useEffect(() => {
    // 初始化花朵
    flowersRef.current = Array.from({ length: calculateFlowerCount() }, createFlower);

    // 开始动画
    lastTimeRef.current = performance.now();
    animationRef.current = requestAnimationFrame(optimizedAnimateWithRender);

    // 监听窗口大小变化
    window.addEventListener('resize', handleResize);

    // 页面可见性检测，避免后台运行时消耗资源
    const handleVisibilityChange = () => {
      isVisible.current = !document.hidden;
      if (isVisible.current && !animationRef.current) {
        lastTimeRef.current = performance.now();
        animationRef.current = requestAnimationFrame(optimizedAnimateWithRender);
      } else if (!isVisible.current && animationRef.current) {
        cancelAnimationFrame(animationRef.current);
        animationRef.current = undefined;
      }
    };

    document.addEventListener('visibilitychange', handleVisibilityChange);

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
      window.removeEventListener('resize', handleResize);
      document.removeEventListener('visibilitychange', handleVisibilityChange);
    };
  }, [optimizedAnimateWithRender, handleResize, calculateFlowerCount, createFlower]);

  return (
    <div ref={containerRef} style={{ position: 'fixed', top: 0, left: 0, width: '100%', height: '100%', pointerEvents: 'none', zIndex: 1 }}>
      {flowersRef.current.map(flower => (
        <RealisticFlower
          key={flower.id}
          x={flower.x}
          y={flower.y}
          size={flower.size}
          flowerType={flower.flowerType}
          rotation={flower.rotation}
          opacity={flower.opacity}
          bloomProgress={flower.bloomProgress}
        />
      ))}
    </div>
  );
};

export default FloatingFlowers;