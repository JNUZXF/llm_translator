/**
 * 主题切换组件
 * 提供优雅的主题切换UI
 */

import React from 'react';
import styled from 'styled-components';
import { useTheme } from '../theme/ThemeContext';

// ============= 样式组件 =============

const ToggleButton = styled.button<{ $isActive?: boolean }>`
  position: relative;
  width: 60px;
  height: 32px;
  border-radius: 16px;
  border: none;
  cursor: pointer;
  background: ${props => props.$isActive
    ? 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)'
    : props.theme.colors.neutral.gray[200]
  };
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  outline: none;
  overflow: hidden;

  &:hover {
    transform: scale(1.05);
    box-shadow: ${props => props.theme.shadows.md};
  }

  &:active {
    transform: scale(0.98);
  }

  &:focus-visible {
    outline: 2px solid ${props => props.theme.colors.border.focus};
    outline-offset: 2px;
  }
`;

const ToggleCircle = styled.div<{ $position: 'left' | 'right' }>`
  position: absolute;
  top: 3px;
  ${props => props.$position === 'left' ? 'left: 3px' : 'right: 3px'};
  width: 26px;
  height: 26px;
  border-radius: 50%;
  background: white;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  display: flex;
  align-items: center;
  justify-content: center;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
`;

const IconWrapper = styled.span`
  font-size: 14px;
  line-height: 1;
  display: flex;
  align-items: center;
  justify-content: center;
`;

// ============= 图标组件 =============

const SunIcon = () => (
  <IconWrapper>
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
      <circle cx="12" cy="12" r="4"/>
      <path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M6.34 17.66l-1.41 1.41M19.07 4.93l-1.41 1.41"/>
    </svg>
  </IconWrapper>
);

const MoonIcon = () => (
  <IconWrapper>
    <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
      <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>
    </svg>
  </IconWrapper>
);

// ============= 主组件 =============

interface ThemeToggleProps {
  className?: string;
  showLabel?: boolean;
}

export const ThemeToggle: React.FC<ThemeToggleProps> = ({ className, showLabel = false }) => {
  const { mode, toggleMode } = useTheme();
  const isDark = mode === 'dark';

  return (
    <Container className={className}>
      {showLabel && <Label>主题</Label>}
      <ToggleButton
        onClick={toggleMode}
        $isActive={isDark}
        aria-label={`切换到${isDark ? '亮色' : '暗色'}主题`}
        title={`当前: ${isDark ? '暗色' : '亮色'}主题`}
      >
        <ToggleCircle $position={isDark ? 'right' : 'left'}>
          {isDark ? <MoonIcon /> : <SunIcon />}
        </ToggleCircle>
      </ToggleButton>
    </Container>
  );
};

const Container = styled.div`
  display: flex;
  align-items: center;
  gap: ${props => props.theme.spacing[2]};
`;

const Label = styled.span`
  font-size: ${props => props.theme.typography.fontSize.sm};
  color: ${props => props.theme.colors.text.secondary};
  font-weight: ${props => props.theme.typography.fontWeight.medium};
`;

// ============= 高级主题切换组件（带下拉菜单） =============

const DropdownContainer = styled.div`
  position: relative;
`;

const DropdownButton = styled.button`
  display: flex;
  align-items: center;
  gap: ${props => props.theme.spacing[2]};
  padding: ${props => props.theme.spacing[2]} ${props => props.theme.spacing[3]};
  border-radius: ${props => props.theme.borderRadius.md};
  border: 1px solid ${props => props.theme.colors.border.default};
  background: ${props => props.theme.colors.background.paper};
  color: ${props => props.theme.colors.text.primary};
  font-size: ${props => props.theme.typography.fontSize.sm};
  font-weight: ${props => props.theme.typography.fontWeight.medium};
  cursor: pointer;
  transition: all ${props => props.theme.transitions.duration.fast} ${props => props.theme.transitions.timing.easeInOut};

  &:hover {
    background: ${props => props.theme.colors.background.elevated};
    border-color: ${props => props.theme.colors.border.dark};
  }

  &:focus-visible {
    outline: 2px solid ${props => props.theme.colors.border.focus};
    outline-offset: 2px;
  }
`;

const DropdownMenu = styled.div<{ $isOpen: boolean }>`
  position: absolute;
  top: calc(100% + ${props => props.theme.spacing[1]});
  right: 0;
  min-width: 150px;
  background: ${props => props.theme.colors.background.paper};
  border: 1px solid ${props => props.theme.colors.border.default};
  border-radius: ${props => props.theme.borderRadius.md};
  box-shadow: ${props => props.theme.shadows.lg};
  opacity: ${props => props.$isOpen ? 1 : 0};
  visibility: ${props => props.$isOpen ? 'visible' : 'hidden'};
  transform: ${props => props.$isOpen ? 'translateY(0)' : 'translateY(-10px)'};
  transition: all ${props => props.theme.transitions.duration.fast} ${props => props.theme.transitions.timing.easeOut};
  z-index: ${props => props.theme.zIndex.dropdown};
`;

const MenuItem = styled.button<{ $isActive?: boolean }>`
  width: 100%;
  display: flex;
  align-items: center;
  gap: ${props => props.theme.spacing[2]};
  padding: ${props => props.theme.spacing[2]} ${props => props.theme.spacing[3]};
  border: none;
  background: ${props => props.$isActive
    ? props.theme.colors.semantic.primary.lighter
    : 'transparent'
  };
  color: ${props => props.$isActive
    ? props.theme.colors.semantic.primary.dark
    : props.theme.colors.text.primary
  };
  font-size: ${props => props.theme.typography.fontSize.sm};
  cursor: pointer;
  transition: all ${props => props.theme.transitions.duration.fast};

  &:hover {
    background: ${props => props.$isActive
      ? props.theme.colors.semantic.primary.lighter
      : props.theme.colors.background.elevated
    };
  }

  &:first-child {
    border-top-left-radius: ${props => props.theme.borderRadius.md};
    border-top-right-radius: ${props => props.theme.borderRadius.md};
  }

  &:last-child {
    border-bottom-left-radius: ${props => props.theme.borderRadius.md};
    border-bottom-right-radius: ${props => props.theme.borderRadius.md};
  }
`;

const CheckIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
    <path d="M20 6L9 17l-5-5"/>
  </svg>
);

interface ThemeDropdownProps {
  className?: string;
}

export const ThemeDropdown: React.FC<ThemeDropdownProps> = ({ className }) => {
  const { mode, currentTheme, availableThemes, setTheme, toggleMode } = useTheme();
  const [isOpen, setIsOpen] = React.useState(false);

  const handleToggle = () => {
    setIsOpen(!isOpen);
  };

  const handleSelect = (themeName: string) => {
    setTheme(themeName);
    setIsOpen(false);
  };

  // 点击外部关闭下拉菜单
  React.useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      const target = event.target as HTMLElement;
      if (!target.closest('[data-theme-dropdown]')) {
        setIsOpen(false);
      }
    };

    if (isOpen) {
      document.addEventListener('click', handleClickOutside);
      return () => document.removeEventListener('click', handleClickOutside);
    }
  }, [isOpen]);

  return (
    <DropdownContainer className={className} data-theme-dropdown>
      <DropdownButton onClick={handleToggle}>
        {mode === 'dark' ? <MoonIcon /> : <SunIcon />}
        <span>{mode === 'dark' ? '暗色' : '亮色'}主题</span>
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M6 9l6 6 6-6"/>
        </svg>
      </DropdownButton>
      <DropdownMenu $isOpen={isOpen}>
        {availableThemes.map(themeName => (
          <MenuItem
            key={themeName}
            onClick={() => handleSelect(themeName)}
            $isActive={currentTheme === themeName}
          >
            {currentTheme === themeName && <CheckIcon />}
            <span>{themeName === 'light' ? '亮色' : themeName === 'dark' ? '暗色' : themeName}</span>
          </MenuItem>
        ))}
      </DropdownMenu>
    </DropdownContainer>
  );
};

export default ThemeToggle;
