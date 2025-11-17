/**
 * 主题化的全局样式
 * 使用主题系统重写的GlobalStyles
 */

import styled, { createGlobalStyle, keyframes, css } from 'styled-components';
import { Theme } from '../theme/types';

// ============= 全局样式 =============

export const GlobalStyle = createGlobalStyle<{ theme: Theme }>`
  * {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
  }

  html {
    font-size: 16px;
    scroll-behavior: smooth;
  }

  body {
    font-family: ${props => props.theme.typography.fontFamily.primary};
    font-size: ${props => props.theme.typography.fontSize.base};
    line-height: ${props => props.theme.typography.lineHeight.normal};
    color: ${props => props.theme.colors.text.primary};
    background: ${props => props.theme.mode === 'light'
      ? 'linear-gradient(135deg, #667eea 0%, #764ba2 25%, #f093fb 50%, #f5576c 75%, #4facfe 100%)'
      : 'linear-gradient(135deg, #1a1a2e 0%, #16213e 25%, #0f3460 50%, #533483 75%, #1a1a3e 100%)'
    };
    background-size: 400% 400%;
    animation: gradientShift 15s ease infinite;
    min-height: 100vh;
    overflow-x: hidden;
    position: relative;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
    transition: background ${props => props.theme.transitions.duration.slow} ${props => props.theme.transitions.timing.easeInOut};
  }

  @keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
  }

  /* 装饰性背景层 */
  body::before {
    content: '';
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: ${props => props.theme.mode === 'light'
      ? `radial-gradient(circle at 20% 80%, rgba(102, 126, 234, 0.3) 0%, transparent 50%),
         radial-gradient(circle at 80% 20%, rgba(240, 147, 251, 0.3) 0%, transparent 50%),
         radial-gradient(circle at 40% 40%, rgba(79, 172, 254, 0.2) 0%, transparent 50%)`
      : `radial-gradient(circle at 20% 80%, rgba(21, 35, 62, 0.8) 0%, transparent 50%),
         radial-gradient(circle at 80% 20%, rgba(83, 52, 131, 0.6) 0%, transparent 50%),
         radial-gradient(circle at 40% 40%, rgba(15, 52, 96, 0.7) 0%, transparent 50%)`
    };
    pointer-events: none;
    z-index: 0;
    transition: background ${props => props.theme.transitions.duration.slow} ${props => props.theme.transitions.timing.easeInOut};
  }

  code {
    font-family: ${props => props.theme.typography.fontFamily.monospace};
  }

  /* 滚动条样式 */
  ::-webkit-scrollbar {
    width: 8px;
    height: 8px;
  }

  ::-webkit-scrollbar-track {
    background: ${props => props.theme.colors.background.paper};
    border-radius: ${props => props.theme.borderRadius.sm};
  }

  ::-webkit-scrollbar-thumb {
    background: ${props => props.theme.colors.neutral.gray[400]};
    border-radius: ${props => props.theme.borderRadius.sm};

    &:hover {
      background: ${props => props.theme.colors.neutral.gray[500]};
    }
  }

  /* 选择文本样式 */
  ::selection {
    background: ${props => props.theme.colors.semantic.primary.light};
    color: ${props => props.theme.colors.semantic.primary.contrast};
  }

  /* 焦点样式 */
  :focus-visible {
    outline: 2px solid ${props => props.theme.colors.border.focus};
    outline-offset: 2px;
  }

  /* 禁用元素样式 */
  :disabled {
    cursor: not-allowed;
    opacity: 0.6;
  }

  /* 动画关键帧 */
  @keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
  }

  @keyframes slideUp {
    from {
      opacity: 0;
      transform: translateY(20px);
    }
    to {
      opacity: 1;
      transform: translateY(0);
    }
  }

  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
  }

  @keyframes rotate {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
  }
`;

// ============= 浮动花朵动画 =============

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
  transition: background ${props => props.theme.transitions.duration.normal};

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

// ============= 布局组件 =============

export const AppContainer = styled.div`
  display: flex;
  min-height: 100vh;
  position: relative;
  z-index: 2;
`;

export const Sidebar = styled.nav`
  width: 240px;
  background: ${props => props.theme.mode === 'light'
    ? 'rgba(255, 255, 255, 0.1)'
    : 'rgba(0, 0, 0, 0.3)'
  };
  backdrop-filter: blur(25px);
  -webkit-backdrop-filter: blur(25px);
  border-right: 1px solid ${props => props.theme.colors.border.default};
  padding: ${props => props.theme.spacing[8]} ${props => props.theme.spacing[4]};
  display: flex;
  flex-direction: column;
  align-items: center;
  box-shadow: ${props => props.theme.shadows.lg};
  position: relative;
  transition: all ${props => props.theme.transitions.duration.normal};

  &::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: linear-gradient(
      180deg,
      ${props => props.theme.mode === 'light'
        ? 'rgba(255, 255, 255, 0.1) 0%, rgba(255, 255, 255, 0.05) 50%, rgba(255, 255, 255, 0.1) 100%'
        : 'rgba(255, 255, 255, 0.02) 0%, rgba(255, 255, 255, 0.01) 50%, rgba(255, 255, 255, 0.02) 100%'
      }
    );
    pointer-events: none;
  }
`;

export const Logo = styled.h1`
  font-size: ${props => props.theme.typography.fontSize['4xl']};
  font-weight: ${props => props.theme.typography.fontWeight.bold};
  margin-bottom: ${props => props.theme.spacing[12]};
  text-align: center;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
  background-size: 200% auto;
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  position: relative;
  z-index: 1;
  filter: drop-shadow(0 4px 8px rgba(102, 126, 234, 0.3));
  animation: ${props => props.theme.animations.fadeIn};

  &::after {
    content: '';
    position: absolute;
    top: 100%;
    left: 50%;
    width: 60px;
    height: 3px;
    background: linear-gradient(135deg, #667eea 0%, #f093fb 100%);
    border-radius: ${props => props.theme.borderRadius.sm};
    transform: translateX(-50%);
    margin-top: ${props => props.theme.spacing[2]};
  }
`;

export const NavButton = styled.button<{ $active?: boolean }>`
  width: 100%;
  padding: ${props => props.theme.spacing[5]} ${props => props.theme.spacing[6]};
  margin: ${props => props.theme.spacing[3]} 0;
  border: none;
  border-radius: ${props => props.theme.borderRadius.xl};
  background: ${props => props.$active
    ? 'linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%)'
    : props.theme.mode === 'light'
      ? 'rgba(255, 255, 255, 0.15)'
      : 'rgba(255, 255, 255, 0.08)'
  };
  color: ${props => props.$active
    ? props.theme.colors.semantic.primary.contrast
    : props.theme.colors.text.primary
  };
  font-size: ${props => props.theme.typography.fontSize.base};
  font-weight: ${props => props.theme.typography.fontWeight.semibold};
  cursor: pointer;
  transition: all ${props => props.theme.transitions.duration.normal} ${props => props.theme.transitions.timing.easeInOut};
  backdrop-filter: blur(10px);
  border: 1px solid ${props => props.$active
    ? 'transparent'
    : props.theme.colors.border.light
  };
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
    transition: left ${props => props.theme.transitions.duration.slow} ease;
  }

  &:hover {
    transform: translateY(-3px) scale(1.02);
    box-shadow: ${props => props.theme.shadows.xl};
    background: ${props => props.$active
      ? 'linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%)'
      : props.theme.mode === 'light'
        ? 'rgba(255, 255, 255, 0.25)'
        : 'rgba(255, 255, 255, 0.12)'
    };
    border-color: ${props => props.theme.colors.border.default};

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
  padding: ${props => props.theme.spacing[8]};
  display: flex;
  flex-direction: column;
  gap: ${props => props.theme.spacing[8]};
  position: relative;
  z-index: 1;
`;

export const ContentCard = styled.div`
  background: ${props => props.theme.colors.background.paper};
  backdrop-filter: blur(10px);
  border-radius: ${props => props.theme.borderRadius['2xl']};
  padding: ${props => props.theme.spacing[8]};
  box-shadow: ${props => props.theme.shadows.xl};
  border: 1px solid ${props => props.theme.colors.border.light};
  flex: 1;
  display: flex;
  flex-direction: column;
  transition: all ${props => props.theme.transitions.duration.normal};
`;

// ============= 表单组件 =============

const inputBaseStyles = css`
  width: 100%;
  padding: ${props => props.theme.spacing[5]};
  border: 2px solid ${props => props.theme.colors.border.default};
  border-radius: ${props => props.theme.borderRadius.lg};
  font-size: ${props => props.theme.typography.fontSize.base};
  line-height: ${props => props.theme.typography.lineHeight.normal};
  outline: none;
  background: ${props => props.theme.colors.background.paper};
  color: ${props => props.theme.colors.text.primary};
  backdrop-filter: blur(5px);
  transition: all ${props => props.theme.transitions.duration.fast} ${props => props.theme.transitions.timing.easeInOut};

  &:focus {
    border-color: ${props => props.theme.colors.border.focus};
    box-shadow: 0 0 0 3px ${props => props.theme.colors.semantic.primary.lighter};
    background: ${props => props.theme.colors.background.elevated};
  }

  &:disabled {
    background: ${props => props.theme.colors.neutral.gray[100]};
    color: ${props => props.theme.colors.text.disabled};
    cursor: not-allowed;
  }

  &::placeholder {
    color: ${props => props.theme.colors.text.hint};
  }
`;

export const TextArea = styled.textarea`
  ${inputBaseStyles}
  min-height: 500px;
  flex: 1;
  resize: vertical;
  overflow-y: auto;
  font-family: ${props => props.theme.typography.fontFamily.primary};
`;

export const LargeTextArea = styled(TextArea)`
  min-height: 700px;
  resize: none;

  &.fast-translation {
    height: 100%;
    min-height: 600px !important;
  }
`;

export const ResultArea = styled.div`
  ${inputBaseStyles}
  min-height: 700px;
  flex: 1;
  overflow-y: auto;
  white-space: pre-wrap;
  word-wrap: break-word;

  &.fast-translation {
    height: 100%;
    min-height: 400px !important;
  }
`;

export const DraftTextArea = styled.textarea`
  ${inputBaseStyles}
  min-height: 120px;
  flex: 1;
  padding: ${props => props.theme.spacing[4]};
  border-color: ${props => props.theme.colors.semantic.success.light};
  background: ${props => props.theme.mode === 'light'
    ? 'rgba(248, 255, 248, 0.8)'
    : props.theme.colors.background.paper
  };

  &:focus {
    border-color: ${props => props.theme.colors.semantic.success.main};
    box-shadow: 0 0 0 3px ${props => props.theme.colors.semantic.success.lighter};
  }
`;

export const RequirementsInput = styled.textarea`
  ${inputBaseStyles}
  height: 80px;
  padding: ${props => props.theme.spacing[3]};
  border-color: ${props => props.theme.colors.semantic.warning.light};
  background: ${props => props.theme.mode === 'light'
    ? 'rgba(255, 248, 240, 0.8)'
    : props.theme.colors.background.paper
  };

  &:focus {
    border-color: ${props => props.theme.colors.semantic.warning.main};
    box-shadow: 0 0 0 3px ${props => props.theme.colors.semantic.warning.lighter};
  }
`;

export const Select = styled.select`
  padding: ${props => props.theme.spacing[3]} ${props => props.theme.spacing[4]};
  border: 2px solid ${props => props.theme.colors.border.default};
  border-radius: ${props => props.theme.borderRadius.md};
  font-size: ${props => props.theme.typography.fontSize.base};
  background: ${props => props.theme.colors.background.paper};
  color: ${props => props.theme.colors.text.primary};
  backdrop-filter: blur(5px);
  outline: none;
  cursor: pointer;
  transition: all ${props => props.theme.transitions.duration.fast};

  &:focus {
    border-color: ${props => props.theme.colors.border.focus};
    box-shadow: 0 0 0 3px ${props => props.theme.colors.semantic.primary.lighter};
  }
`;

export const Button = styled.button`
  padding: ${props => props.theme.spacing[3]} ${props => props.theme.spacing[6]};
  border: none;
  border-radius: ${props => props.theme.borderRadius.md};
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: ${props => props.theme.colors.semantic.primary.contrast};
  font-size: ${props => props.theme.typography.fontSize.base};
  font-weight: ${props => props.theme.typography.fontWeight.medium};
  cursor: pointer;
  transition: all ${props => props.theme.transitions.duration.fast};

  &:hover {
    transform: translateY(-2px);
    box-shadow: ${props => props.theme.shadows.lg};
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

// ============= 其他组件 =============

export const FileUploadArea = styled.div<{ $isDragOver?: boolean }>`
  border: 2px dashed ${props => props.$isDragOver
    ? props.theme.colors.border.focus
    : props.theme.colors.border.default
  };
  border-radius: ${props => props.theme.borderRadius.xl};
  padding: ${props => props.theme.spacing[12]};
  text-align: center;
  background: ${props => props.$isDragOver
    ? props.theme.colors.semantic.primary.lighter
    : props.theme.colors.background.paper
  };
  backdrop-filter: blur(5px);
  transition: all ${props => props.theme.transitions.duration.fast};
  cursor: pointer;

  &:hover {
    border-color: ${props => props.theme.colors.border.focus};
    background: ${props => props.theme.colors.semantic.primary.lighter};
  }
`;

export const ErrorMessage = styled.div`
  background: ${props => props.theme.colors.semantic.error.lighter};
  border: 1px solid ${props => props.theme.colors.semantic.error.light};
  border-radius: ${props => props.theme.borderRadius.md};
  padding: ${props => props.theme.spacing[4]};
  color: ${props => props.theme.colors.semantic.error.dark};
  margin: ${props => props.theme.spacing[4]} 0;
`;

export const LoadingSpinner = styled.div`
  width: 20px;
  height: 20px;
  border: 2px solid ${props => props.theme.colors.neutral.gray[300]};
  border-top: 2px solid ${props => props.theme.colors.semantic.primary.main};
  border-radius: 50%;
  animation: spin 1s linear infinite;
  display: inline-block;
  margin-right: ${props => props.theme.spacing[2]};

  @keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
  }
`;

export const FlexRow = styled.div<{ $align?: string }>`
  display: flex;
  gap: ${props => props.theme.spacing[4]};
  align-items: ${props => props.$align || 'center'};
  margin-bottom: ${props => props.theme.spacing[4]};
`;

export const FlexColumn = styled.div`
  display: flex;
  flex-direction: column;
  gap: ${props => props.theme.spacing[4]};
  flex: 1;
`;

export const Label = styled.label`
  font-weight: ${props => props.theme.typography.fontWeight.medium};
  color: ${props => props.theme.colors.semantic.primary.main};
  margin-bottom: ${props => props.theme.spacing[2]};
  display: block;
`;
