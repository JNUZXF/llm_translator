import React, { useState, useEffect } from 'react';
import { AppContainer, Sidebar, Logo, NavButton, MainContent } from './styles/GlobalStyles';
import ThemedGlobalStyles from './styles/ThemedGlobalStyles';
import { ThemeProvider } from './theme';
import { ThemeToggle } from './components/ThemeToggle';
import ParticleBackground from './components/ParticleBackground';
import FloatingFlowers from './components/FloatingFlowers';
import HomePage from './pages/HomePage';
import FastTranslation from './pages/FastTranslation';
import PaperTranslation from './pages/PaperTranslation';
import { usePageState } from './hooks/usePersistedState';
import { useAppSettings } from './hooks/useAppSettings';

type Page = 'home' | 'fast' | 'paper';

const App: React.FC = () => {
  const [currentPage, setCurrentPage] = usePageState();
  const [showSettings, setShowSettings] = useState(false);
  const { clearAllData, exportData } = useAppSettings();

  const handleNavigateFromHome = (page: 'fast' | 'paper') => {
    setCurrentPage(page);
  };

  const handleClearData = () => {
    if (window.confirm('确定要清除所有保存的数据吗？这将重置应用到初始状态。')) {
      clearAllData();
    }
  };

  const handleExportData = () => {
    const data = exportData();
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'ai-translator-settings.json';
    a.click();
    URL.revokeObjectURL(url);
    setShowSettings(false);
  };

  // 点击外部关闭设置菜单
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (showSettings) {
        const target = event.target as HTMLElement;
        if (!target.closest('[data-settings-menu]')) {
          setShowSettings(false);
        }
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [showSettings]);

  const renderCurrentPage = () => {
    switch (currentPage) {
      case 'home':
        return <HomePage onNavigate={handleNavigateFromHome} />;
      case 'fast':
        return <FastTranslation />;
      case 'paper':
        return <PaperTranslation />;
      default:
        return <HomePage onNavigate={handleNavigateFromHome} />;
    }
  };

  return (
    <ThemeProvider>
      <ThemedGlobalStyles />
      <ParticleBackground />
      <FloatingFlowers />
      <AppContainer>
        <Sidebar style={{ justifyContent: 'space-between' }}>
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', width: '100%' }}>
            <Logo>AI翻译助手</Logo>
            <NavButton
              $active={currentPage === 'home'}
              onClick={() => setCurrentPage('home')}
            >
              🏠 首页
            </NavButton>
            <NavButton
              $active={currentPage === 'fast'}
              onClick={() => setCurrentPage('fast')}
            >
              ⚡ 快速翻译
            </NavButton>
            <NavButton
              $active={currentPage === 'paper'}
              onClick={() => setCurrentPage('paper')}
            >
              📄 论文翻译
            </NavButton>
          </div>
          
          <div style={{ position: 'relative', width: '100%' }} data-settings-menu>
            {/* 主题切换器 */}
            <div style={{
              padding: '0.75rem 1rem',
              borderTop: '1px solid rgba(255, 255, 255, 0.1)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center'
            }}>
              <ThemeToggle />
            </div>

            <NavButton
              $active={false}
              onClick={() => setShowSettings(!showSettings)}
            >
              ⚙️ 设置
            </NavButton>
            
            {showSettings && (
              <div data-settings-menu style={{
                position: 'absolute',
                bottom: '100%',
                left: '1rem',
                right: '1rem',
                backgroundColor: 'rgba(255, 255, 255, 0.95)',
                borderRadius: '8px',
                padding: '0.75rem',
                marginBottom: '0.5rem',
                boxShadow: '0 4px 12px rgba(0, 0, 0, 0.15)',
                backdropFilter: 'blur(10px)',
                border: '1px solid rgba(255, 255, 255, 0.2)'
              }}>
                <div style={{ fontSize: '0.9rem', marginBottom: '0.75rem', color: '#666', fontWeight: 'bold' }}>
                  数据管理
                </div>
                <button
                  onClick={handleExportData}
                  style={{
                    width: '100%',
                    padding: '0.75rem',
                    marginBottom: '0.5rem',
                    border: 'none',
                    borderRadius: '6px',
                    backgroundColor: '#e3f2fd',
                    color: '#1976d2',
                    cursor: 'pointer',
                    fontSize: '0.9rem',
                    fontWeight: '500',
                    transition: 'background-color 0.2s'
                  }}
                  onMouseOver={(e) => (e.target as HTMLButtonElement).style.backgroundColor = '#bbdefb'}
                  onMouseOut={(e) => (e.target as HTMLButtonElement).style.backgroundColor = '#e3f2fd'}
                >
                  📤 导出设置
                </button>
                <button
                  onClick={handleClearData}
                  style={{
                    width: '100%',
                    padding: '0.75rem',
                    border: 'none',
                    borderRadius: '6px',
                    backgroundColor: '#ffebee',
                    color: '#d32f2f',
                    cursor: 'pointer',
                    fontSize: '0.9rem',
                    fontWeight: '500',
                    transition: 'background-color 0.2s'
                  }}
                  onMouseOver={(e) => (e.target as HTMLButtonElement).style.backgroundColor = '#ffcdd2'}
                  onMouseOut={(e) => (e.target as HTMLButtonElement).style.backgroundColor = '#ffebee'}
                >
                  🗑️ 清除所有数据
                </button>
              </div>
            )}
          </div>
        </Sidebar>
        
        <MainContent>
          {renderCurrentPage()}
        </MainContent>
      </AppContainer>
    </ThemeProvider>
  );
};

export default App;