import { useState, useEffect } from 'react';

/**
 * 自定义钩子：持久化状态到localStorage
 * @param key - localStorage中的键名
 * @param defaultValue - 默认值
 * @returns [state, setState] - 状态和设置状态的函数
 */
export function usePersistedState<T>(
  key: string,
  defaultValue: T
): [T, React.Dispatch<React.SetStateAction<T>>] {
  // 从localStorage读取初始值
  const [state, setState] = useState<T>(() => {
    try {
      const item = window.localStorage.getItem(key);
      return item ? JSON.parse(item) : defaultValue;
    } catch (error) {
      console.warn(`Error reading localStorage key "${key}":`, error);
      return defaultValue;
    }
  });

  // 当状态改变时，保存到localStorage
  useEffect(() => {
    try {
      window.localStorage.setItem(key, JSON.stringify(state));
    } catch (error) {
      console.warn(`Error setting localStorage key "${key}":`, error);
    }
  }, [key, state]);

  return [state, setState];
}

/**
 * 专门用于页面状态的钩子
 */
export function usePageState() {
  return usePersistedState<'home' | 'fast' | 'paper'>('ai-translator-current-page', 'home');
}
