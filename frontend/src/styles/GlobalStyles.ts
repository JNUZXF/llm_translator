import styled, { createGlobalStyle, keyframes } from 'styled-components';

export const GlobalStyle = createGlobalStyle`
  * {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
  }

  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen',
      'Ubuntu', 'Cantarell', 'Fira Sans', 'Droid Sans', 'Helvetica Neue',
      sans-serif;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
    background: linear-gradient(
      135deg,
      #667eea 0%,
      #764ba2 25%,
      #f093fb 50%,
      #f5576c 75%,
      #4facfe 100%
    );
    background-size: 400% 400%;
    animation: gradientShift 15s ease infinite;
    min-height: 100vh;
    overflow-x: hidden;
    position: relative;
  }
  
  @keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
  }
  
  body::before {
    content: '';
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: 
      radial-gradient(circle at 20% 80%, rgba(102, 126, 234, 0.3) 0%, transparent 50%),
      radial-gradient(circle at 80% 20%, rgba(240, 147, 251, 0.3) 0%, transparent 50%),
      radial-gradient(circle at 40% 40%, rgba(79, 172, 254, 0.2) 0%, transparent 50%);
    pointer-events: none;
    z-index: 0;
  }

  code {
    font-family: source-code-pro, Menlo, Monaco, Consolas, 'Courier New',
      monospace;
  }

  ::-webkit-scrollbar {
    width: 8px;
  }

  ::-webkit-scrollbar-track {
    background: rgba(255, 255, 255, 0.1);
    border-radius: 4px;
  }

  ::-webkit-scrollbar-thumb {
    background: rgba(255, 255, 255, 0.3);
    border-radius: 4px;
  }

  ::-webkit-scrollbar-thumb:hover {
    background: rgba(255, 255, 255, 0.5);
  }
`;

const float = keyframes`
  0%, 100% { transform: translateY(0px) rotate(0deg); }
  50% { transform: translateY(-20px) rotate(180deg); }
`;

export const FloatingFlower = styled.div<{
  $x: number;
  $y: number;
  $size: number;
  $color: string;
  $duration: number;
}>`
  position: fixed;
  left: ${props => props.$x}px;
  top: ${props => props.$y}px;
  width: ${props => props.$size}px;
  height: ${props => props.$size}px;
  background: ${props => props.$color};
  border-radius: 50% 0 50% 0;
  animation: ${float} ${props => props.$duration}s ease-in-out infinite;
  pointer-events: none;
  z-index: 1;
  opacity: 0.7;
  
  &::before {
    content: '';
    position: absolute;
    width: 100%;
    height: 100%;
    background: inherit;
    border-radius: 0 50% 0 50%;
    transform: rotate(45deg);
  }
`;

export const AppContainer = styled.div`
  display: flex;
  min-height: 100vh;
  position: relative;
  z-index: 2;
`;

export const Sidebar = styled.nav`
  width: 240px;
  background: rgba(255, 255, 255, 0.1);
  backdrop-filter: blur(25px);
  border-right: 1px solid rgba(255, 255, 255, 0.2);
  padding: 2rem 1rem; /* 增加左右内边距，避免按钮贴边 */
  display: flex;
  flex-direction: column;
  align-items: center;
  box-shadow: 
    2px 0 30px rgba(0, 0, 0, 0.1),
    inset -1px 0 0 rgba(255, 255, 255, 0.1);
  position: relative;
  
  &::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: linear-gradient(
      180deg,
      rgba(255, 255, 255, 0.1) 0%,
      rgba(255, 255, 255, 0.05) 50%,
      rgba(255, 255, 255, 0.1) 100%
    );
    pointer-events: none;
  }
`;

export const Logo = styled.h1`
  font-size: 1.8rem;
  font-weight: 700;
  color: #667eea;
  margin-bottom: 3rem;
  text-align: center;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
  background-size: 200% auto;
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  position: relative;
  z-index: 1;
  filter: drop-shadow(0 4px 8px rgba(102, 126, 234, 0.3));
  
  &::after {
    content: '';
    position: absolute;
    top: 100%;
    left: 50%;
    width: 60px;
    height: 3px;
    background: linear-gradient(135deg, #667eea 0%, #f093fb 100%);
    border-radius: 2px;
    transform: translateX(-50%);
    margin-top: 0.5rem;
  }
`;

export const NavButton = styled.button<{ $active?: boolean }>`
  width: 100%;
  padding: 1.2rem 1.5rem;
  margin: 0.8rem 0;
  border: none;
  border-radius: 16px;
  background: ${props => props.$active 
    ? 'linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%)' 
    : 'rgba(255, 255, 255, 0.15)'};
  color: ${props => props.$active ? 'white' : 'rgba(255, 255, 255, 0.9)'};
  font-size: 1rem;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
  backdrop-filter: blur(10px);
  border: 1px solid ${props => props.$active ? 'transparent' : 'rgba(255, 255, 255, 0.3)'};
  position: relative;
  z-index: 1;
  overflow: hidden;
  text-shadow: ${props => props.$active ? '0 2px 8px rgba(0, 0, 0, 0.3)' : 'none'};
  
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
      rgba(255, 255, 255, 0.2),
      transparent
    );
    transition: left 0.6s ease;
  }

  &:hover {
    transform: translateY(-3px) scale(1.02);
    box-shadow: 0 12px 30px rgba(102, 126, 234, 0.4);
    background: ${props => props.$active 
      ? 'linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%)' 
      : 'rgba(255, 255, 255, 0.25)'};
    border-color: rgba(255, 255, 255, 0.5);
    
    &::before {
      left: 100%;
    }
  }

  &:active {
    transform: translateY(-1px) scale(1.01);
  }
`;

export const MainContent = styled.main`
  flex: 1;
  padding: 2rem;
  display: flex;
  flex-direction: column;
  gap: 2rem;
  position: relative;
  z-index: 1;
`;

export const ContentCard = styled.div`
  background: rgba(255, 255, 255, 0.95);
  backdrop-filter: blur(10px);
  border-radius: 20px;
  padding: 2rem;
  box-shadow: 0 10px 40px rgba(0, 0, 0, 0.1);
  border: 1px solid rgba(255, 255, 255, 0.2);
  flex: 1;
  display: flex;
  flex-direction: column;
`;

// Base TextArea with flexible height
export const TextArea = styled.textarea`
  width: 100%;
  min-height: 500px;
  flex: 1;
  padding: 1.5rem;
  border: 2px solid rgba(102, 126, 234, 0.2);
  border-radius: 12px;
  font-size: 1rem;
  line-height: 1.6;
  resize: vertical;
  outline: none;
  background: rgba(255, 255, 255, 0.8);
  backdrop-filter: blur(5px);
  transition: all 0.3s ease;
  overflow-y: auto;

  &:focus {
    border-color: #667eea;
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    background: rgba(255, 255, 255, 0.95);
  }

  &:disabled {
    background: rgba(0, 0, 0, 0.05);
    color: #666;
    cursor: not-allowed;
  }

  &::placeholder {
    color: #999;
  }
`;

// Base ResultArea with flexible height
export const ResultArea = styled.div`
  width: 100%;
  min-height: 700px;
  flex: 1;
  padding: 1.5rem;
  border: 2px solid rgba(102, 126, 234, 0.2);
  border-radius: 12px;
  font-size: 1rem;
  line-height: 1.6;
  background: rgba(255, 255, 255, 0.8);
  backdrop-filter: blur(5px);
  overflow-y: auto;
  white-space: pre-wrap;
  color: #333;
`;

// Large flexible TextArea for main translation input
export const LargeTextArea = styled(TextArea)`
  min-height: 700px;
  resize: none;
  
  // For fast translation page, ensure sufficient height
  &.fast-translation {
    height: 100%;
    min-height: 600px !important; // 输入框高度 - 在这里改！
  }
`;

// Large flexible ResultArea for main translation output
export const LargeResultArea = styled(ResultArea)`
  min-height: 700px;
  
  // For fast translation page, ensure sufficient height
  &.fast-translation {
    height: 100%;
    min-height: 400px !important; // 输出框高度 - 在这里改！
  }
`;

// Draft TextArea with flexible height
export const DraftTextArea = styled.textarea`
  width: 100%;
  min-height: 120px;
  flex: 1;
  padding: 1rem;
  border: 2px solid rgba(139, 195, 74, 0.3);
  border-radius: 8px;
  font-size: 0.9rem;
  line-height: 1.5;
  resize: vertical;
  outline: none;
  background: rgba(248, 255, 248, 0.8);
  backdrop-filter: blur(5px);
  transition: all 0.3s ease;
  overflow-y: auto;
  
  &:focus {
    border-color: #8bc34a;
    box-shadow: 0 0 0 3px rgba(139, 195, 74, 0.1);
    background: rgba(248, 255, 248, 0.95);
  }

  &::placeholder {
    color: #999;
  }
`;

// 额外要求输入框样式
export const RequirementsInput = styled.textarea`
  width: 100%;
  height: 80px;
  padding: 0.75rem;
  border: 2px solid rgba(255, 152, 0, 0.3);
  border-radius: 8px;
  font-size: 0.9rem;
  line-height: 1.4;
  resize: vertical;
  outline: none;
  background: rgba(255, 248, 240, 0.8);
  backdrop-filter: blur(5px);
  transition: all 0.3s ease;
  overflow-y: auto;
  
  &:focus {
    border-color: #ff9800;
    box-shadow: 0 0 0 3px rgba(255, 152, 0, 0.1);
    background: rgba(255, 248, 240, 0.95);
  }

  &::placeholder {
    color: #999;
  }
`;

export const Select = styled.select`
  padding: 0.75rem 1rem;
  border: 2px solid rgba(102, 126, 234, 0.2);
  border-radius: 8px;
  font-size: 1rem;
  background: rgba(255, 255, 255, 0.9);
  backdrop-filter: blur(5px);
  outline: none;
  cursor: pointer;
  transition: all 0.3s ease;

  &:focus {
    border-color: #667eea;
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
  }
`;

export const Button = styled.button`
  padding: 0.75rem 1.5rem;
  border: none;
  border-radius: 8px;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  font-size: 1rem;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.3s ease;

  &:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
  }

  &:active {
    transform: translateY(0);
  }

  &:disabled {
    opacity: 0.6;
    cursor: not-allowed;
    transform: none;
  }
`;

export const FileUploadArea = styled.div<{ $isDragOver?: boolean }>`
  border: 2px dashed ${props => props.$isDragOver ? '#667eea' : 'rgba(102, 126, 234, 0.3)'};
  border-radius: 12px;
  padding: 3rem;
  text-align: center;
  background: ${props => props.$isDragOver 
    ? 'rgba(102, 126, 234, 0.05)' 
    : 'rgba(255, 255, 255, 0.5)'};
  backdrop-filter: blur(5px);
  transition: all 0.3s ease;
  cursor: pointer;

  &:hover {
    border-color: #667eea;
    background: rgba(102, 126, 234, 0.05);
  }
`;

export const ErrorMessage = styled.div`
  background: rgba(255, 99, 99, 0.1);
  border: 1px solid rgba(255, 99, 99, 0.3);
  border-radius: 8px;
  padding: 1rem;
  color: #d32f2f;
  margin: 1rem 0;
`;

export const LoadingSpinner = styled.div`
  width: 20px;
  height: 20px;
  border: 2px solid rgba(255, 255, 255, 0.3);
  border-top: 2px solid white;
  border-radius: 50%;
  animation: spin 1s linear infinite;
  display: inline-block;
  margin-right: 0.5rem;

  @keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
  }
`;

export const FlexRow = styled.div<{ $align?: string }>`
  display: flex;
  gap: 1rem;
  align-items: ${props => props.$align || 'center'};
  margin-bottom: 1rem;
`;

export const FlexColumn = styled.div`
  display: flex;
  flex-direction: column;
  gap: 1rem;
  flex: 1;
`;

export const Label = styled.label`
  font-weight: 500;
  color: #667eea;
  margin-bottom: 0.5rem;
  display: block;
`;