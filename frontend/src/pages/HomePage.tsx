import React from 'react';
import styled, { keyframes } from 'styled-components';

const fadeInUp = keyframes`
  from {
    opacity: 0;
    transform: translateY(50px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
`;

const floating = keyframes`
  0%, 100% { transform: translateY(0px); }
  50% { transform: translateY(-20px); }
`;

const shimmer = keyframes`
  0% { background-position: -200px 0; }
  100% { background-position: calc(200px + 100%) 0; }
`;

const HomeContainer = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  min-height: 100vh;
  padding: 2rem;
  text-align: center;
  position: relative;
`;

const Hero = styled.div`
  max-width: 800px;
  margin-bottom: 4rem;
  animation: ${fadeInUp} 1s ease-out;
`;

const MainTitle = styled.h1`
  font-size: 4.5rem;
  font-weight: 800;
  margin-bottom: 1.5rem;
  background: linear-gradient(
    135deg,
    #667eea 0%,
    #764ba2 25%,
    #f093fb 50%,
    #f5576c 75%,
    #4facfe 100%
  );
  background-size: 200% auto;
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  animation: ${shimmer} 3s ease-in-out infinite;
  text-shadow: 0 4px 20px rgba(102, 126, 234, 0.3);
  
  @media (max-width: 768px) {
    font-size: 3rem;
  }
`;

const Subtitle = styled.h2`
  font-size: 1.8rem;
  font-weight: 300;
  color: rgba(255, 255, 255, 0.9);
  margin-bottom: 2rem;
  line-height: 1.4;
  text-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
  animation: ${fadeInUp} 1s ease-out 0.3s both;
`;

const Description = styled.p`
  font-size: 1.2rem;
  color: rgba(255, 255, 255, 0.8);
  line-height: 1.8;
  margin-bottom: 3rem;
  animation: ${fadeInUp} 1s ease-out 0.6s both;
`;

const FeatureGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 2rem;
  width: 100%;
  max-width: 1200px;
  margin-bottom: 4rem;
`;

const FeatureCard = styled.div`
  background: rgba(255, 255, 255, 0.1);
  backdrop-filter: blur(20px);
  border-radius: 20px;
  padding: 2.5rem;
  border: 1px solid rgba(255, 255, 255, 0.2);
  box-shadow: 
    0 20px 40px rgba(0, 0, 0, 0.1),
    inset 0 1px 0 rgba(255, 255, 255, 0.2);
  transition: all 0.4s ease;
  animation: ${fadeInUp} 1s ease-out 0.9s both;
  position: relative;
  overflow: hidden;
  
  &:hover {
    transform: translateY(-10px) scale(1.02);
    box-shadow: 
      0 30px 60px rgba(0, 0, 0, 0.2),
      inset 0 1px 0 rgba(255, 255, 255, 0.3);
    background: rgba(255, 255, 255, 0.15);
  }
  
  &::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(
      90deg,
      transparent,
      rgba(255, 255, 255, 0.1),
      transparent
    );
    transition: left 0.8s ease;
  }
  
  &:hover::before {
    left: 100%;
  }
`;

const FeatureIcon = styled.div`
  font-size: 3rem;
  margin-bottom: 1.5rem;
  animation: ${floating} 3s ease-in-out infinite;
  
  &:nth-child(odd) {
    animation-delay: -1.5s;
  }
`;

const FeatureTitle = styled.h3`
  font-size: 1.5rem;
  font-weight: 600;
  color: white;
  margin-bottom: 1rem;
  text-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
`;

const FeatureDescription = styled.p`
  color: rgba(255, 255, 255, 0.8);
  line-height: 1.6;
  font-size: 1rem;
`;

const CTASection = styled.div`
  animation: ${fadeInUp} 1s ease-out 1.2s both;
`;

const StartButton = styled.button`
  padding: 1.2rem 3rem;
  font-size: 1.2rem;
  font-weight: 600;
  color: white;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  border: none;
  border-radius: 50px;
  cursor: pointer;
  transition: all 0.4s ease;
  position: relative;
  overflow: hidden;
  box-shadow: 
    0 10px 30px rgba(102, 126, 234, 0.4),
    inset 0 1px 0 rgba(255, 255, 255, 0.2);
  
  &:hover {
    transform: translateY(-3px) scale(1.05);
    box-shadow: 
      0 15px 40px rgba(102, 126, 234, 0.6),
      inset 0 1px 0 rgba(255, 255, 255, 0.3);
  }
  
  &:active {
    transform: translateY(-1px) scale(1.02);
  }
  
  &::before {
    content: '';
    position: absolute;
    top: 50%;
    left: 50%;
    width: 0;
    height: 0;
    border-radius: 50%;
    background: rgba(255, 255, 255, 0.2);
    transform: translate(-50%, -50%);
    transition: width 0.6s ease, height 0.6s ease;
  }
  
  &:hover::before {
    width: 300px;
    height: 300px;
  }
  
  span {
    position: relative;
    z-index: 1;
  }
`;

const StatCard = styled.div`
  display: inline-block;
  margin: 0 1rem;
  padding: 1.5rem;
  background: rgba(255, 255, 255, 0.1);
  backdrop-filter: blur(10px);
  border-radius: 15px;
  border: 1px solid rgba(255, 255, 255, 0.2);
  animation: ${fadeInUp} 1s ease-out 1.5s both;
  
  h4 {
    font-size: 2rem;
    font-weight: 700;
    color: #f093fb;
    margin-bottom: 0.5rem;
    text-shadow: 0 2px 8px rgba(240, 147, 251, 0.5);
  }
  
  p {
    color: rgba(255, 255, 255, 0.8);
    font-size: 0.9rem;
    margin: 0;
  }
`;

interface HomePageProps {
  onNavigate?: (page: 'fast' | 'paper') => void;
}

const HomePage: React.FC<HomePageProps> = ({ onNavigate }) => {
  const handleStartClick = () => {
    if (onNavigate) {
      onNavigate('fast');
    }
  };

  return (
    <HomeContainer>
      <Hero>
        <MainTitle>Better Translation</MainTitle>
        <Subtitle>AI驱动的智能翻译平台</Subtitle>
        <Description>
          体验前所未有的翻译质量，支持22种语言的实时翻译和专业论文翻译服务。
          让语言不再成为沟通的障碍，让世界更加紧密相连。
        </Description>
      </Hero>

      <FeatureGrid>
        <FeatureCard>
          <FeatureIcon>⚡</FeatureIcon>
          <FeatureTitle>闪电般的速度</FeatureTitle>
          <FeatureDescription>
            基于最新AI技术，实现毫秒级响应速度，
            流式输出让您实时看到翻译过程，体验丝滑般流畅。
          </FeatureDescription>
        </FeatureCard>

        <FeatureCard>
          <FeatureIcon>🎯</FeatureIcon>
          <FeatureTitle>精准如人工</FeatureTitle>
          <FeatureDescription>
            多模型融合技术，上下文理解能力强，
            翻译准确率高达98%，保持原文语意和语调。
          </FeatureDescription>
        </FeatureCard>

        <FeatureCard>
          <FeatureIcon>📚</FeatureIcon>
          <FeatureTitle>学术级专业</FeatureTitle>
          <FeatureDescription>
            专门优化的学术论文翻译模式，
            保持专业术语准确性，格式完美呈现。
          </FeatureDescription>
        </FeatureCard>

        <FeatureCard>
          <FeatureIcon>🌍</FeatureIcon>
          <FeatureTitle>全球化支持</FeatureTitle>
          <FeatureDescription>
            支持22种主流语言互译，
            覆盖全球95%的语言使用场景。
          </FeatureDescription>
        </FeatureCard>

        <FeatureCard>
          <FeatureIcon>🔒</FeatureIcon>
          <FeatureTitle>隐私保护</FeatureTitle>
          <FeatureDescription>
            端到端加密传输，不存储您的翻译内容，
            严格遵循数据保护法规。
          </FeatureDescription>
        </FeatureCard>

        <FeatureCard>
          <FeatureIcon>💎</FeatureIcon>
          <FeatureTitle>极致体验</FeatureTitle>
          <FeatureDescription>
            精美的界面设计，流畅的交互动画，
            让翻译变成一种享受。
          </FeatureDescription>
        </FeatureCard>
      </FeatureGrid>

      <div style={{ marginBottom: '3rem' }}>
        <StatCard>
          <h4>22+</h4>
          <p>支持语言</p>
        </StatCard>
        <StatCard>
          <h4>98%</h4>
          <p>翻译准确率</p>
        </StatCard>
        <StatCard>
          <h4>0.5s</h4>
          <p>平均响应时间</p>
        </StatCard>
        <StatCard>
          <h4>100k+</h4>
          <p>每日翻译量</p>
        </StatCard>
      </div>

      <CTASection>
        <StartButton onClick={handleStartClick}>
          <span>开始使用</span>
        </StartButton>
      </CTASection>
    </HomeContainer>
  );
};

export default HomePage;
