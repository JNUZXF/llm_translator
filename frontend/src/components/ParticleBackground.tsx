import React, { useEffect, useRef } from 'react';
import styled from 'styled-components';

const Canvas = styled.canvas`
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  pointer-events: none;
  z-index: 0;
  opacity: 0.6;
`;

interface Particle {
  x: number;
  y: number;
  vx: number;
  vy: number;
  size: number;
  opacity: number;
  color: string;
  life: number;
  maxLife: number;
}

const ParticleBackground: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const particlesRef = useRef<Particle[]>([]);
  const animationRef = useRef<number>();

  const colors = [
    'rgba(102, 126, 234, 0.8)',
    'rgba(118, 75, 162, 0.8)',
    'rgba(240, 147, 251, 0.8)',
    'rgba(245, 87, 108, 0.8)',
    'rgba(79, 172, 254, 0.8)',
  ];

  const createParticle = (canvas: HTMLCanvasElement): Particle => {
    return {
      x: Math.random() * canvas.width,
      y: Math.random() * canvas.height,
      vx: (Math.random() - 0.5) * 0.5,
      vy: (Math.random() - 0.5) * 0.5,
      size: Math.random() * 3 + 1,
      opacity: Math.random() * 0.5 + 0.3,
      color: colors[Math.floor(Math.random() * colors.length)],
      life: 0,
      maxLife: Math.random() * 300 + 200,
    };
  };

  const updateParticle = (particle: Particle, canvas: HTMLCanvasElement) => {
    particle.x += particle.vx;
    particle.y += particle.vy;
    particle.life++;

    // 边界检测
    if (particle.x < 0 || particle.x > canvas.width) particle.vx *= -1;
    if (particle.y < 0 || particle.y > canvas.height) particle.vy *= -1;

    // 生命周期管理
    const lifeRatio = particle.life / particle.maxLife;
    particle.opacity = (1 - lifeRatio) * 0.5;

    // 重生
    if (particle.life >= particle.maxLife) {
      Object.assign(particle, createParticle(canvas));
    }
  };

  const drawParticle = (ctx: CanvasRenderingContext2D, particle: Particle) => {
    ctx.save();
    ctx.globalAlpha = particle.opacity;
    
    // 绘制发光效果
    const gradient = ctx.createRadialGradient(
      particle.x, particle.y, 0,
      particle.x, particle.y, particle.size * 2
    );
    gradient.addColorStop(0, particle.color);
    gradient.addColorStop(1, 'transparent');
    
    ctx.fillStyle = gradient;
    ctx.beginPath();
    ctx.arc(particle.x, particle.y, particle.size * 2, 0, Math.PI * 2);
    ctx.fill();
    
    // 绘制核心
    ctx.fillStyle = particle.color;
    ctx.beginPath();
    ctx.arc(particle.x, particle.y, particle.size, 0, Math.PI * 2);
    ctx.fill();
    
    ctx.restore();
  };

  const connectParticles = (ctx: CanvasRenderingContext2D, particles: Particle[]) => {
    for (let i = 0; i < particles.length; i++) {
      for (let j = i + 1; j < particles.length; j++) {
        const dx = particles[i].x - particles[j].x;
        const dy = particles[i].y - particles[j].y;
        const distance = Math.sqrt(dx * dx + dy * dy);

        if (distance < 100) {
          ctx.save();
          ctx.globalAlpha = (1 - distance / 100) * 0.2;
          ctx.strokeStyle = 'rgba(255, 255, 255, 0.5)';
          ctx.lineWidth = 0.5;
          ctx.beginPath();
          ctx.moveTo(particles[i].x, particles[i].y);
          ctx.lineTo(particles[j].x, particles[j].y);
          ctx.stroke();
          ctx.restore();
        }
      }
    }
  };

  const animate = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // 清除画布
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // 更新和绘制粒子
    particlesRef.current.forEach(particle => {
      updateParticle(particle, canvas);
      drawParticle(ctx, particle);
    });

    // 连接邻近粒子
    connectParticles(ctx, particlesRef.current);

    animationRef.current = requestAnimationFrame(animate);
  };

  const handleResize = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;

    // 重新初始化粒子
    particlesRef.current = Array.from({ length: 80 }, () => createParticle(canvas));
  };

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    // 初始化画布
    handleResize();

    // 开始动画
    animate();

    // 添加窗口大小变化监听
    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, []);

  return <Canvas ref={canvasRef} />;
};

export default ParticleBackground;
