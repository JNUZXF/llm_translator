/**
 * 主题上下文
 * 提供主题切换功能和主题数据访问
 */

import React, { createContext, useContext, useState, useEffect, useMemo, ReactNode } from 'react';
import { Theme, ThemeMode, ThemeContextType, ThemeOptions } from './types';
import { themes } from './themes';

// 创建Context
const ThemeContext = createContext<ThemeContextType | undefined>(undefined);

// Provider Props
interface ThemeProviderProps {
  children: ReactNode;
  options?: ThemeOptions;
}

// 默认配置
const defaultOptions: Required<ThemeOptions> = {
  defaultMode: 'light',
  persist: true,
  storageKey: 'ai-translator-theme',
  customThemes: {},
};

/**
 * 主题Provider组件
 */
export function ThemeProvider({ children, options = {} }: ThemeProviderProps) {
  const config = { ...defaultOptions, ...options };
  const allThemes = { ...themes, ...config.customThemes };

  // 从localStorage读取保存的主题模式
  const getInitialMode = (): ThemeMode => {
    if (!config.persist) {
      return config.defaultMode;
    }

    try {
      const stored = localStorage.getItem(config.storageKey);
      if (stored && (stored === 'light' || stored === 'dark')) {
        return stored as ThemeMode;
      }
    } catch (error) {
      console.warn('Failed to read theme from localStorage:', error);
    }

    // 检测系统偏好
    if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
      return 'dark';
    }

    return config.defaultMode;
  };

  const [mode, setModeState] = useState<ThemeMode>(getInitialMode);
  const [currentThemeName, setCurrentThemeName] = useState<string>(mode);

  // 保存主题模式到localStorage
  const persistMode = (newMode: ThemeMode) => {
    if (!config.persist) return;

    try {
      localStorage.setItem(config.storageKey, newMode);
    } catch (error) {
      console.warn('Failed to save theme to localStorage:', error);
    }
  };

  // 设置主题模式
  const setMode = (newMode: ThemeMode) => {
    setModeState(newMode);
    setCurrentThemeName(newMode);
    persistMode(newMode);

    // 更新document的data-theme属性（用于CSS变量）
    document.documentElement.setAttribute('data-theme', newMode);
  };

  // 切换主题模式
  const toggleMode = () => {
    const newMode = mode === 'light' ? 'dark' : 'light';
    setMode(newMode);
  };

  // 设置特定主题
  const setTheme = (themeName: string) => {
    if (!allThemes[themeName]) {
      console.warn(`Theme "${themeName}" not found`);
      return;
    }

    const theme = allThemes[themeName];
    setModeState(theme.mode);
    setCurrentThemeName(themeName);
    persistMode(theme.mode);

    // 更新document属性
    document.documentElement.setAttribute('data-theme', theme.mode);
    document.documentElement.setAttribute('data-theme-name', themeName);
  };

  // 获取当前主题
  const theme = useMemo<Theme>(() => {
    return allThemes[currentThemeName] || allThemes[mode];
  }, [currentThemeName, mode, allThemes]);

  // 监听系统主题变化
  useEffect(() => {
    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');

    const handleChange = (e: MediaQueryListEvent) => {
      if (!config.persist || !localStorage.getItem(config.storageKey)) {
        const newMode = e.matches ? 'dark' : 'light';
        setMode(newMode);
      }
    };

    // 现代浏览器使用addEventListener
    if (mediaQuery.addEventListener) {
      mediaQuery.addEventListener('change', handleChange);
      return () => mediaQuery.removeEventListener('change', handleChange);
    }
    // 旧浏览器使用addListener (deprecated)
    else {
      mediaQuery.addListener(handleChange);
      return () => mediaQuery.removeListener(handleChange);
    }
  }, [config.persist, config.storageKey]);

  // 初始化时设置document属性
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', mode);
    document.documentElement.setAttribute('data-theme-name', currentThemeName);
  }, [mode, currentThemeName]);

  const value: ThemeContextType = {
    theme,
    mode,
    setMode,
    toggleMode,
    availableThemes: Object.keys(allThemes),
    currentTheme: currentThemeName,
    setTheme,
  };

  return (
    <ThemeContext.Provider value={value}>
      {children}
    </ThemeContext.Provider>
  );
}

/**
 * 使用主题Hook
 * @throws {Error} 如果在ThemeProvider外部使用
 */
export function useTheme(): ThemeContextType {
  const context = useContext(ThemeContext);

  if (context === undefined) {
    throw new Error('useTheme must be used within a ThemeProvider');
  }

  return context;
}

/**
 * 使用主题数据Hook（不包含切换功能）
 */
export function useThemeData(): Theme {
  const { theme } = useTheme();
  return theme;
}

/**
 * 使用主题模式Hook
 */
export function useThemeMode(): [ThemeMode, (mode: ThemeMode) => void, () => void] {
  const { mode, setMode, toggleMode } = useTheme();
  return [mode, setMode, toggleMode];
}
