/**
 * 主题配置
 * 定义了亮色和暗色主题，以及主题工厂函数
 */

import { Theme, Palette } from './types';
import { typography, spacing, shadows, borderRadius, transitions, animations, zIndex, breakpoints } from './tokens';

// ============= 辅助函数 =============

/**
 * 创建调色板
 */
function createPalette(
  main: string,
  light: string,
  lighter: string,
  dark: string,
  darker: string,
  contrast: string
): Palette {
  return { main, light, lighter, dark, darker, contrast };
}

// ============= 亮色主题 =============

export const lightTheme: Theme = {
  mode: 'light',
  name: 'Light',
  colors: {
    semantic: {
      primary: createPalette(
        '#6366f1',    // Indigo 500
        '#818cf8',    // Indigo 400
        '#a5b4fc',    // Indigo 300
        '#4f46e5',    // Indigo 600
        '#4338ca',    // Indigo 700
        '#ffffff'
      ),
      secondary: createPalette(
        '#ec4899',    // Pink 500
        '#f472b6',    // Pink 400
        '#f9a8d4',    // Pink 300
        '#db2777',    // Pink 600
        '#be185d',    // Pink 700
        '#ffffff'
      ),
      success: createPalette(
        '#10b981',    // Emerald 500
        '#34d399',    // Emerald 400
        '#6ee7b7',    // Emerald 300
        '#059669',    // Emerald 600
        '#047857',    // Emerald 700
        '#ffffff'
      ),
      warning: createPalette(
        '#f59e0b',    // Amber 500
        '#fbbf24',    // Amber 400
        '#fcd34d',    // Amber 300
        '#d97706',    // Amber 600
        '#b45309',    // Amber 700
        '#ffffff'
      ),
      error: createPalette(
        '#ef4444',    // Red 500
        '#f87171',    // Red 400
        '#fca5a5',    // Red 300
        '#dc2626',    // Red 600
        '#b91c1c',    // Red 700
        '#ffffff'
      ),
      info: createPalette(
        '#3b82f6',    // Blue 500
        '#60a5fa',    // Blue 400
        '#93c5fd',    // Blue 300
        '#2563eb',    // Blue 600
        '#1d4ed8',    // Blue 700
        '#ffffff'
      ),
    },
    neutral: {
      white: '#ffffff',
      black: '#000000',
      gray: {
        50: '#f9fafb',
        100: '#f3f4f6',
        200: '#e5e7eb',
        300: '#d1d5db',
        400: '#9ca3af',
        500: '#6b7280',
        600: '#4b5563',
        700: '#374151',
        800: '#1f2937',
        900: '#111827',
      },
    },
    background: {
      default: '#ffffff',
      paper: '#f9fafb',
      elevated: '#ffffff',
      overlay: 'rgba(0, 0, 0, 0.5)',
    },
    text: {
      primary: '#111827',      // Gray 900
      secondary: '#6b7280',    // Gray 500
      disabled: '#9ca3af',     // Gray 400
      hint: '#d1d5db',         // Gray 300
      inverse: '#ffffff',
    },
    border: {
      default: '#e5e7eb',      // Gray 200
      light: '#f3f4f6',        // Gray 100
      dark: '#d1d5db',         // Gray 300
      focus: '#6366f1',        // Primary main
    },
  },
  typography,
  spacing,
  shadows,
  borderRadius,
  transitions,
  animations,
  zIndex,
  breakpoints,
};

// ============= 暗色主题 =============

export const darkTheme: Theme = {
  mode: 'dark',
  name: 'Dark',
  colors: {
    semantic: {
      primary: createPalette(
        '#818cf8',    // Indigo 400
        '#a5b4fc',    // Indigo 300
        '#c7d2fe',    // Indigo 200
        '#6366f1',    // Indigo 500
        '#4f46e5',    // Indigo 600
        '#000000'
      ),
      secondary: createPalette(
        '#f472b6',    // Pink 400
        '#f9a8d4',    // Pink 300
        '#fbcfe8',    // Pink 200
        '#ec4899',    // Pink 500
        '#db2777',    // Pink 600
        '#000000'
      ),
      success: createPalette(
        '#34d399',    // Emerald 400
        '#6ee7b7',    // Emerald 300
        '#a7f3d0',    // Emerald 200
        '#10b981',    // Emerald 500
        '#059669',    // Emerald 600
        '#000000'
      ),
      warning: createPalette(
        '#fbbf24',    // Amber 400
        '#fcd34d',    // Amber 300
        '#fde68a',    // Amber 200
        '#f59e0b',    // Amber 500
        '#d97706',    // Amber 600
        '#000000'
      ),
      error: createPalette(
        '#f87171',    // Red 400
        '#fca5a5',    // Red 300
        '#fecaca',    // Red 200
        '#ef4444',    // Red 500
        '#dc2626',    // Red 600
        '#000000'
      ),
      info: createPalette(
        '#60a5fa',    // Blue 400
        '#93c5fd',    // Blue 300
        '#bfdbfe',    // Blue 200
        '#3b82f6',    // Blue 500
        '#2563eb',    // Blue 600
        '#000000'
      ),
    },
    neutral: {
      white: '#ffffff',
      black: '#000000',
      gray: {
        50: '#18181b',   // Zinc 900
        100: '#27272a',  // Zinc 800
        200: '#3f3f46',  // Zinc 700
        300: '#52525b',  // Zinc 600
        400: '#71717a',  // Zinc 500
        500: '#a1a1aa',  // Zinc 400
        600: '#d4d4d8',  // Zinc 300
        700: '#e4e4e7',  // Zinc 200
        800: '#f4f4f5',  // Zinc 100
        900: '#fafafa',  // Zinc 50
      },
    },
    background: {
      default: '#0f172a',      // Slate 900
      paper: '#1e293b',        // Slate 800
      elevated: '#334155',     // Slate 700
      overlay: 'rgba(0, 0, 0, 0.75)',
    },
    text: {
      primary: '#f1f5f9',      // Slate 100
      secondary: '#94a3b8',    // Slate 400
      disabled: '#64748b',     // Slate 500
      hint: '#475569',         // Slate 600
      inverse: '#0f172a',      // Slate 900
    },
    border: {
      default: '#334155',      // Slate 700
      light: '#475569',        // Slate 600
      dark: '#1e293b',         // Slate 800
      focus: '#818cf8',        // Primary main
    },
  },
  typography,
  spacing,
  shadows: {
    ...shadows,
    // 暗色主题的阴影需要更深
    xs: '0 1px 2px 0 rgba(0, 0, 0, 0.3)',
    sm: '0 1px 3px 0 rgba(0, 0, 0, 0.4), 0 1px 2px 0 rgba(0, 0, 0, 0.3)',
    md: '0 4px 6px -1px rgba(0, 0, 0, 0.4), 0 2px 4px -1px rgba(0, 0, 0, 0.3)',
    lg: '0 10px 15px -3px rgba(0, 0, 0, 0.4), 0 4px 6px -2px rgba(0, 0, 0, 0.3)',
    xl: '0 20px 25px -5px rgba(0, 0, 0, 0.4), 0 10px 10px -5px rgba(0, 0, 0, 0.3)',
    '2xl': '0 25px 50px -12px rgba(0, 0, 0, 0.5)',
  },
  borderRadius,
  transitions,
  animations,
  zIndex,
  breakpoints,
};

// ============= 主题映射 =============

export const themes: Record<string, Theme> = {
  light: lightTheme,
  dark: darkTheme,
};

// ============= 主题工厂 =============

/**
 * 创建自定义主题
 * 允许基于现有主题进行扩展
 */
export function createTheme(baseTheme: Theme, overrides: Partial<Theme>): Theme {
  return {
    ...baseTheme,
    ...overrides,
    colors: {
      ...baseTheme.colors,
      ...overrides.colors,
    },
  };
}

/**
 * 合并多个主题
 */
export function mergeThemes(...themesList: Partial<Theme>[]): Partial<Theme> {
  return themesList.reduce((acc, theme) => ({
    ...acc,
    ...theme,
    colors: {
      ...acc.colors,
      ...theme.colors,
    },
  }), {});
}
