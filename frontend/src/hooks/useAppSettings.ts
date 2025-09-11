import { useCallback } from 'react';

/**
 * 应用设置管理钩子
 */
export function useAppSettings() {
  // 清除所有应用数据
  const clearAllData = useCallback(() => {
    const keysToRemove = [
      'ai-translator-current-page',
      'fast-translation-input',
      'fast-translation-language',
      'fast-translation-scene',
      'fast-translation-requirements',
      'fast-translation-draft',
      'paper-translation-selected-page',
      'paper-translation-file-info'
    ];

    keysToRemove.forEach(key => {
      try {
        localStorage.removeItem(key);
      } catch (error) {
        console.warn(`Error removing localStorage key "${key}":`, error);
      }
    });

    // 刷新页面以重置所有状态
    window.location.reload();
  }, []);

  // 导出应用数据（用于备份）
  const exportData = useCallback(() => {
    const data: Record<string, any> = {};
    const keys = [
      'ai-translator-current-page',
      'fast-translation-input',
      'fast-translation-language',
      'fast-translation-scene',
      'fast-translation-requirements',
      'fast-translation-draft',
      'paper-translation-selected-page',
      'paper-translation-file-info'
    ];

    keys.forEach(key => {
      try {
        const value = localStorage.getItem(key);
        if (value) {
          data[key] = JSON.parse(value);
        }
      } catch (error) {
        console.warn(`Error reading localStorage key "${key}":`, error);
      }
    });

    return data;
  }, []);

  // 导入应用数据（用于恢复）
  const importData = useCallback((data: Record<string, any>) => {
    Object.entries(data).forEach(([key, value]) => {
      try {
        localStorage.setItem(key, JSON.stringify(value));
      } catch (error) {
        console.warn(`Error setting localStorage key "${key}":`, error);
      }
    });

    // 刷新页面以应用新状态
    window.location.reload();
  }, []);

  return {
    clearAllData,
    exportData,
    importData
  };
}
