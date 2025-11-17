# 🎨 主题系统完整指南

## 目录
- [概述](#概述)
- [快速开始](#快速开始)
- [架构设计](#架构设计)
- [使用指南](#使用指南)
- [主题扩展](#主题扩展)
- [最佳实践](#最佳实践)
- [API参考](#api参考)

---

## 概述

这是一个生产级别的主题管理系统，具有以下特点：

### ✨ 核心特性

- **🎯 类型安全**：完整的TypeScript类型定义
- **🌗 双主题支持**：内置亮色和暗色主题
- **🔧 完全可扩展**：轻松添加自定义主题
- **💾 持久化**：自动保存用户偏好到localStorage
- **📱 响应式**：跟随系统主题偏好
- **🎨 设计令牌**：基于设计令牌的一致性系统
- **⚡ 性能优化**：使用Context API和useMemo优化
- **🎭 优雅切换**：流畅的主题切换动画

### 📦 文件结构

```
frontend/src/
├── theme/
│   ├── types.ts          # TypeScript类型定义
│   ├── tokens.ts         # 设计令牌（间距、颜色、字体等）
│   ├── themes.ts         # 主题配置（亮色、暗色）
│   ├── ThemeContext.tsx  # Context和Hooks
│   └── index.ts          # 统一导出
├── components/
│   └── ThemeToggle.tsx   # 主题切换组件
└── styles/
    └── ThemedGlobalStyles.ts  # 主题化的全局样式
```

---

## 快速开始

### 1. 安装主题系统

主题系统已经内置，无需额外安装。

### 2. 在应用中启用主题

```tsx
// src/App.tsx
import React from 'react';
import { ThemeProvider } from './theme';
import { GlobalStyle } from './styles/ThemedGlobalStyles';
import { ThemeToggle } from './components/ThemeToggle';

function App() {
  return (
    <ThemeProvider>
      <GlobalStyle />
      <div>
        <ThemeToggle />
        {/* 你的应用内容 */}
      </div>
    </ThemeProvider>
  );
}

export default App;
```

### 3. 使用主题

```tsx
import React from 'react';
import styled from 'styled-components';
import { useTheme } from './theme';

const MyComponent = () => {
  const { theme } = useTheme();

  return (
    <Container>
      <Title>当前主题：{theme.mode}</Title>
    </Container>
  );
};

const Container = styled.div`
  background: ${props => props.theme.colors.background.paper};
  color: ${props => props.theme.colors.text.primary};
  padding: ${props => props.theme.spacing[4]};
  border-radius: ${props => props.theme.borderRadius.lg};
`;

const Title = styled.h2`
  font-size: ${props => props.theme.typography.fontSize['2xl']};
  font-weight: ${props => props.theme.typography.fontWeight.bold};
`;
```

---

## 架构设计

### 设计原则

1. **关注点分离**：类型、令牌、主题、逻辑分离
2. **单一数据源**：所有设计决策来自设计令牌
3. **易于扩展**：添加新主题只需几行代码
4. **类型安全**：完整的TypeScript支持
5. **性能优先**：最小化不必要的重渲染

### 层次结构

```
设计令牌（Tokens）
    ↓
主题配置（Themes）
    ↓
主题上下文（Context）
    ↓
组件样式（Styled Components）
```

### 主题对象结构

```typescript
{
  mode: 'light' | 'dark',
  name: string,
  colors: {
    semantic: {     // 语义化颜色
      primary, secondary, success, warning, error, info
    },
    neutral: {      // 中性色
      white, black, gray[50-900]
    },
    background: {   // 背景色
      default, paper, elevated, overlay
    },
    text: {         // 文本色
      primary, secondary, disabled, hint, inverse
    },
    border: {       // 边框色
      default, light, dark, focus
    }
  },
  typography: {     // 排版
    fontFamily, fontSize, fontWeight, lineHeight, letterSpacing
  },
  spacing: {},      // 间距
  shadows: {},      // 阴影
  borderRadius: {}, // 圆角
  transitions: {},  // 过渡
  animations: {},   // 动画
  zIndex: {},       // 层级
  breakpoints: {}   // 断点
}
```

---

## 使用指南

### 基础用法

#### 1. 获取主题数据

```tsx
import { useTheme, useThemeData } from './theme';

// 方式1：获取完整上下文
const { theme, mode, setMode, toggleMode } = useTheme();

// 方式2：仅获取主题数据
const theme = useThemeData();

// 方式3：仅获取主题模式
const [mode, setMode, toggleMode] = useThemeMode();
```

#### 2. 切换主题

```tsx
import { useTheme } from './theme';

const MyComponent = () => {
  const { mode, setMode, toggleMode } = useTheme();

  return (
    <>
      <button onClick={toggleMode}>
        切换主题
      </button>
      <button onClick={() => setMode('light')}>
        亮色主题
      </button>
      <button onClick={() => setMode('dark')}>
        暗色主题
      </button>
    </>
  );
};
```

#### 3. 使用主题切换组件

```tsx
import { ThemeToggle, ThemeDropdown } from './theme';

// 简单切换按钮
<ThemeToggle />

// 带标签的切换按钮
<ThemeToggle showLabel />

// 下拉菜单选择
<ThemeDropdown />
```

### 高级用法

#### 1. 在styled-components中使用

```tsx
import styled from 'styled-components';

const Card = styled.div`
  /* 使用颜色 */
  background: ${props => props.theme.colors.background.paper};
  color: ${props => props.theme.colors.text.primary};
  border: 1px solid ${props => props.theme.colors.border.default};

  /* 使用间距 */
  padding: ${props => props.theme.spacing[4]};
  margin: ${props => props.theme.spacing[2]};

  /* 使用圆角 */
  border-radius: ${props => props.theme.borderRadius.lg};

  /* 使用阴影 */
  box-shadow: ${props => props.theme.shadows.md};

  /* 使用过渡 */
  transition: all ${props => props.theme.transitions.duration.normal}
              ${props => props.theme.transitions.timing.easeInOut};

  /* 使用排版 */
  font-family: ${props => props.theme.typography.fontFamily.primary};
  font-size: ${props => props.theme.typography.fontSize.base};
  font-weight: ${props => props.theme.typography.fontWeight.medium};
  line-height: ${props => props.theme.typography.lineHeight.normal};

  /* 根据主题模式条件渲染 */
  opacity: ${props => props.theme.mode === 'dark' ? 0.9 : 1};

  /* 响应式断点 */
  @media (min-width: ${props => props.theme.breakpoints.md}) {
    padding: ${props => props.theme.spacing[8]};
  }
`;
```

#### 2. 条件样式

```tsx
const Button = styled.button<{ $variant?: 'primary' | 'secondary' }>`
  ${props => {
    const variant = props.$variant || 'primary';
    const colors = props.theme.colors.semantic[variant];

    return `
      background: ${colors.main};
      color: ${colors.contrast};
      border: 2px solid ${colors.main};

      &:hover {
        background: ${colors.dark};
      }

      &:active {
        background: ${colors.darker};
      }
    `;
  }}
`;
```

#### 3. 动态主题值

```tsx
const Component = styled.div<{ $intensity?: number }>`
  background: ${props => {
    const alpha = props.$intensity || 0.5;
    const gray = props.theme.colors.neutral.gray;
    return props.theme.mode === 'light'
      ? `rgba(${gray[900]}, ${alpha})`
      : `rgba(${gray[50]}, ${alpha})`;
  }};
`;
```

---

## 主题扩展

### 创建自定义主题

#### 方式1：基于现有主题扩展

```tsx
import { createTheme, lightTheme } from './theme';

const myCustomTheme = createTheme(lightTheme, {
  name: 'My Custom Theme',
  colors: {
    semantic: {
      ...lightTheme.colors.semantic,
      primary: {
        main: '#ff6b6b',
        light: '#ff8787',
        lighter: '#ffa3a3',
        dark: '#ee5656',
        darker: '#dc4646',
        contrast: '#ffffff',
      },
    },
  },
});
```

#### 方式2：完全自定义

```tsx
import { Theme } from './theme/types';
import { typography, spacing, /* ... */ } from './theme/tokens';

const oceanTheme: Theme = {
  mode: 'light',
  name: 'Ocean',
  colors: {
    semantic: {
      primary: {
        main: '#0077be',
        light: '#0095e8',
        lighter: '#00b4ff',
        dark: '#005a92',
        darker: '#003f66',
        contrast: '#ffffff',
      },
      // ... 其他颜色
    },
    neutral: {
      // ... 中性色
    },
    background: {
      default: '#e6f3ff',
      paper: '#ffffff',
      elevated: '#f0f8ff',
      overlay: 'rgba(0, 119, 190, 0.5)',
    },
    text: {
      primary: '#003f66',
      secondary: '#005a92',
      disabled: '#7fb3d5',
      hint: '#a8d5e8',
      inverse: '#ffffff',
    },
    border: {
      default: '#7fb3d5',
      light: '#a8d5e8',
      dark: '#5a8fb4',
      focus: '#0077be',
    },
  },
  typography,
  spacing,
  // ... 其他令牌
};
```

#### 方式3：注册自定义主题

```tsx
// App.tsx
import { ThemeProvider } from './theme';
import { oceanTheme } from './themes/ocean';

function App() {
  return (
    <ThemeProvider
      options={{
        customThemes: {
          ocean: oceanTheme,
          // 可以添加更多主题
        },
      }}
    >
      {/* 应用内容 */}
    </ThemeProvider>
  );
}
```

#### 使用自定义主题

```tsx
import { useTheme } from './theme';

const ThemeSelector = () => {
  const { currentTheme, availableThemes, setTheme } = useTheme();

  return (
    <select
      value={currentTheme}
      onChange={e => setTheme(e.target.value)}
    >
      {availableThemes.map(themeName => (
        <option key={themeName} value={themeName}>
          {themeName}
        </option>
      ))}
    </select>
  );
};
```

### 创建主题变体

```tsx
// themes/variants.ts
import { mergeThemes, lightTheme, darkTheme } from '../theme';

// 高对比度主题
export const highContrastLight = mergeThemes(lightTheme, {
  colors: {
    text: {
      primary: '#000000',
      secondary: '#333333',
    },
    border: {
      default: '#000000',
    },
  },
});

// 柔和主题
export const softDark = mergeThemes(darkTheme, {
  colors: {
    background: {
      default: '#1a1a1a',
      paper: '#2a2a2a',
    },
  },
});
```

---

## 最佳实践

### 1. 始终使用设计令牌

❌ **不好的做法**：
```tsx
const Component = styled.div`
  padding: 16px;
  margin: 8px;
  color: #333;
`;
```

✅ **好的做法**：
```tsx
const Component = styled.div`
  padding: ${props => props.theme.spacing[4]};
  margin: ${props => props.theme.spacing[2]};
  color: ${props => props.theme.colors.text.primary};
`;
```

### 2. 使用语义化颜色

❌ **不好的做法**：
```tsx
const Button = styled.button`
  background: ${props => props.theme.colors.neutral.gray[500]};
`;
```

✅ **好的做法**：
```tsx
const Button = styled.button`
  background: ${props => props.theme.colors.semantic.primary.main};
  /* 或 */
  background: ${props => props.theme.colors.background.elevated};
`;
```

### 3. 响应式设计

```tsx
const Container = styled.div`
  padding: ${props => props.theme.spacing[2]};

  @media (min-width: ${props => props.theme.breakpoints.sm}) {
    padding: ${props => props.theme.spacing[4]};
  }

  @media (min-width: ${props => props.theme.breakpoints.lg}) {
    padding: ${props => props.theme.spacing[8]};
  }
`;
```

### 4. 可访问性

```tsx
const Link = styled.a`
  color: ${props => props.theme.colors.semantic.primary.main};

  &:focus-visible {
    outline: 2px solid ${props => props.theme.colors.border.focus};
    outline-offset: 2px;
  }

  /* 确保对比度 */
  ${props => {
    const bgColor = props.theme.colors.background.default;
    const textColor = props.theme.colors.text.primary;
    // 可以添加对比度检查逻辑
  }}
`;
```

### 5. 性能优化

```tsx
// ✅ 在组件外部定义样式
const StyledComponent = styled.div`
  /* 样式 */
`;

function MyComponent() {
  return <StyledComponent />;
}

// ❌ 避免在组件内部定义
function MyComponent() {
  const StyledComponent = styled.div`
    /* 每次渲染都会重新创建 */
  `;
  return <StyledComponent />;
}
```

---

## API参考

### ThemeProvider

```tsx
interface ThemeProviderProps {
  children: ReactNode;
  options?: ThemeOptions;
}

interface ThemeOptions {
  defaultMode?: 'light' | 'dark';
  persist?: boolean;
  storageKey?: string;
  customThemes?: Record<string, Partial<Theme>>;
}
```

### useTheme

```tsx
function useTheme(): ThemeContextType

interface ThemeContextType {
  theme: Theme;
  mode: ThemeMode;
  setMode: (mode: ThemeMode) => void;
  toggleMode: () => void;
  availableThemes: string[];
  currentTheme: string;
  setTheme: (themeName: string) => void;
}
```

### useThemeData

```tsx
function useThemeData(): Theme
```

### useThemeMode

```tsx
function useThemeMode(): [
  ThemeMode,
  (mode: ThemeMode) => void,
  () => void
]
```

### 主题工具函数

```tsx
// 创建主题
function createTheme(
  baseTheme: Theme,
  overrides: Partial<Theme>
): Theme

// 合并主题
function mergeThemes(
  ...themes: Partial<Theme>[]
): Partial<Theme>
```

---

## 常见问题

### Q: 如何添加新的颜色？

A: 扩展主题配置：

```tsx
const myTheme = createTheme(lightTheme, {
  colors: {
    custom: {
      brandBlue: '#0066cc',
      brandGreen: '#00cc66',
    },
  },
});
```

### Q: 如何持久化用户的主题选择？

A: ThemeProvider默认启用持久化，保存到localStorage。

### Q: 如何在主题之间共享样式？

A: 使用设计令牌和工具函数：

```tsx
import { typography, spacing } from './theme/tokens';

const sharedTheme = {
  typography,
  spacing,
};

const theme1 = { ...sharedTheme, /* ... */ };
const theme2 = { ...sharedTheme, /* ... */ };
```

### Q: 如何支持更多主题？

A: 创建新主题并注册：

```tsx
<ThemeProvider
  options={{
    customThemes: {
      theme1: myTheme1,
      theme2: myTheme2,
      theme3: myTheme3,
    },
  }}
>
```

---

## 示例项目

查看 `src/examples/ThemeDemo.tsx` 获取完整示例。

---

## 贡献

如果你想为主题系统做贡献，请：

1. Fork项目
2. 创建特性分支
3. 提交你的更改
4. 发起Pull Request

---

## 许可证

MIT License

---

**享受你的主题系统！** 🎨✨
