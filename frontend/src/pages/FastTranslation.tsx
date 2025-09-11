import React, { useState, useEffect } from 'react';
import { 
  ContentCard, 
  LargeTextArea, 
  LargeResultArea, 
  DraftTextArea,
  RequirementsInput,
  Select, 
  FlexRow, 
  FlexColumn, 
  Label,
  ErrorMessage,
  LoadingSpinner,
  Button
} from '../styles/GlobalStyles';
import { api } from '../utils/api';
import { Language, TranslationScene, TranslationResponse } from '../types';
import { usePersistedState } from '../hooks/usePersistedState';

const FastTranslation: React.FC = () => {
  const [inputText, setInputText] = usePersistedState('fast-translation-input', '');
  const [outputText, setOutputText] = useState('');
  const [selectedLanguage, setSelectedLanguage] = usePersistedState('fast-translation-language', 'Deutsch');
  const [selectedScene, setSelectedScene] = usePersistedState('fast-translation-scene', 'ecommerce_amazon');
  const [requirements, setRequirements] = usePersistedState('fast-translation-requirements', '');
  const [draftText, setDraftText] = usePersistedState('fast-translation-draft', '');
  const [languages, setLanguages] = useState<Language[]>([]);
  const [scenes, setScenes] = useState<TranslationScene[]>([]);
  const [isTranslating, setIsTranslating] = useState(false);
  const [isStopping, setIsStopping] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [eventSource, setEventSource] = useState<EventSource | null>(null);
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null);
  
  // 用于跟踪是否是初始化阶段，避免初始化时自动翻译
  const [isInitialized, setIsInitialized] = useState(false);
  // 用于跟踪用户是否主动改变了输入文本
  const [userModifiedInput, setUserModifiedInput] = useState(false);

  // 中断翻译
  const handleStopTranslation = async () => {
    if (isStopping) return; // 防止重复点击
    
    setIsStopping(true);
    try {
      // 如果有会话ID，调用后端取消API
      if (currentSessionId) {
        await api.cancelTranslation(currentSessionId);
        setCurrentSessionId(null);
      }
      
      // 关闭前端连接
      if (eventSource) {
        eventSource.close();
        setEventSource(null);
      }
      
      setIsTranslating(false);
      setError(null);
    } catch (error) {
      console.error('停止翻译失败:', error);
      // 即使后端取消失败，也要停止前端
      if (eventSource) {
        eventSource.close();
        setEventSource(null);
      }
      setIsTranslating(false);
      setCurrentSessionId(null);
    } finally {
      setIsStopping(false);
    }
  };

  const handleTranslate = async () => {
    if (!inputText.trim()) {
      setError('请输入要翻译的文本');
      return;
    }

    setIsTranslating(true);
    setIsStopping(false);
    setError(null);
    setOutputText('');

    try {
      // 关闭之前的连接
      if (eventSource) {
        eventSource.close();
      }

      const es = await api.translateText(inputText, selectedLanguage, selectedScene, requirements);
      setEventSource(es);

      es.onmessage = (event) => {
        try {
          const data: TranslationResponse & { 
            session_id?: string; 
            cancelled?: boolean; 
            message?: string; 
          } = JSON.parse(event.data);
          
          // 处理会话ID
          if (data.session_id) {
            setCurrentSessionId(data.session_id);
            return;
          }
          
          // 处理取消消息
          if (data.cancelled) {
            setError(data.message || '翻译已被中断');
            setIsTranslating(false);
            setIsStopping(false);
            setCurrentSessionId(null);
            es.close();
            return;
          }
          
          if (data.error) {
            setError(data.error);
            setIsTranslating(false);
            setIsStopping(false);
            setCurrentSessionId(null);
            es.close();
            return;
          }

          if (data.content) {
            setOutputText(prev => prev + data.content);
          }

          if (data.done) {
            setIsTranslating(false);
            setIsStopping(false);
            setCurrentSessionId(null);
            setUserModifiedInput(false); // 翻译完成后重置标志
            es.close();
          }
        } catch (err) {
          console.error('解析响应数据失败:', err);
          setError('解析响应数据失败');
          setIsTranslating(false);
          setIsStopping(false);
          setCurrentSessionId(null);
          es.close();
        }
      };

      es.onerror = (error) => {
        console.error('EventSource error:', error);
        setError('翻译服务连接失败，请检查后端服务是否正常运行');
        setIsTranslating(false);
        setIsStopping(false);
        setCurrentSessionId(null);
        es.close();
      };

    } catch (error) {
      console.error('翻译失败:', error);
      setError(error instanceof Error ? error.message : '翻译失败');
      setIsTranslating(false);
      setIsStopping(false);
      setCurrentSessionId(null);
    }
  };

  // 初始化数据
  useEffect(() => {
    const loadData = async () => {
      try {
        const [languagesData, scenesData] = await Promise.all([
          api.getLanguages(),
          api.getScenes()
        ]);
        setLanguages(languagesData);
        setScenes(scenesData);
        
        // 数据加载完成后，标记为已初始化
        setTimeout(() => {
          setIsInitialized(true);
        }, 100); // 给一个小延迟确保所有状态都已设置
      } catch (error) {
        console.error('加载数据失败:', error);
        setError('加载配置数据失败');
        // 即使加载失败也要标记为已初始化
        setTimeout(() => {
          setIsInitialized(true);
        }, 100);
      }
    };

    loadData();
  }, []);

  // 清理EventSource连接
  useEffect(() => {
    return () => {
      if (eventSource) {
        eventSource.close();
      }
      if (currentSessionId) {
        // 组件卸载时尝试取消会话（可选，因为后端有超时清理）
        api.cancelTranslation(currentSessionId).catch(console.error);
      }
    };
  }, [eventSource, currentSessionId]);

  // 当输入文本改变时自动翻译（仅在特定条件下）
  useEffect(() => {
    // 只有在以下情况才自动翻译：
    // 1. 组件已完全初始化
    // 2. 用户主动修改了输入文本，或者切换了语言/场景
    // 3. 有输入文本且当前没在翻译
    if (!isInitialized) return;
    
    const timeoutId = setTimeout(() => {
      if (inputText.trim() && !isTranslating && userModifiedInput) {
        handleTranslate();
      }
    }, 500); // 500ms 防抖

    return () => clearTimeout(timeoutId);
  }, [inputText, selectedLanguage, selectedScene, requirements, isInitialized, userModifiedInput]);

  // 处理用户输入文本的变化
  const handleInputTextChange = (value: string) => {
    setInputText(value);
    if (isInitialized) {
      setUserModifiedInput(true);
    }
  };

  // 处理粘贴事件
  const handlePaste = () => {
    // 粘贴后稍微延迟设置标志，确保文本已更新
    setTimeout(() => {
      if (isInitialized) {
        setUserModifiedInput(true);
      }
    }, 10);
  };

  // 处理语言选择变化
  const handleLanguageChange = (language: string) => {
    setSelectedLanguage(language);
    if (isInitialized && inputText.trim()) {
      setUserModifiedInput(true);
    }
  };

  // 处理场景选择变化
  const handleSceneChange = (scene: string) => {
    setSelectedScene(scene);
    if (isInitialized && inputText.trim()) {
      setUserModifiedInput(true);
    }
  };

  // 处理额外要求变化
  const handleRequirementsChange = (reqs: string) => {
    setRequirements(reqs);
    if (isInitialized && inputText.trim()) {
      setUserModifiedInput(true);
    }
  };

  // 草稿快速操作
  const copyToDraft = () => {
    if (outputText) {
      setDraftText(prev => prev ? `${prev}\n\n--- 翻译结果 ---\n${outputText}` : outputText);
    }
  };

  const copyDraftToInput = () => {
    if (draftText) {
      setInputText(draftText);
      // 从草稿复制到输入框也算用户主动操作
      if (isInitialized) {
        setUserModifiedInput(true);
      }
    }
  };

  const clearDraft = () => {
    if (window.confirm('确定要清空草稿内容吗？')) {
      setDraftText('');
    }
  };

  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      {/* 主要翻译区域 */}
      <FlexRow style={{ height: '70%', gap: '2rem' }}>
        <FlexColumn>
          <ContentCard style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
            <Label>输入文本</Label>
            <LargeTextArea
              className="fast-translation"
              value={inputText}
              onChange={(e) => handleInputTextChange(e.target.value)}
              onPaste={handlePaste}
              placeholder="请输入要翻译的文本..."
              disabled={isTranslating}
            />
          </ContentCard>
        </FlexColumn>

        <FlexColumn>
          <ContentCard style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
            {/* 配置区域 */}
            <div style={{ marginBottom: '1rem' }}>
              <FlexRow style={{ marginBottom: '0.75rem', gap: '1rem' }}>
                <div style={{ flex: 1 }}>
                  <Label style={{ marginBottom: '0.5rem' }}>目标语言</Label>
                  <Select
                    value={selectedLanguage}
                    onChange={(e) => handleLanguageChange(e.target.value)}
                    disabled={isTranslating}
                    style={{ width: '100%' }}
                  >
                    {languages.map((lang) => (
                      <option key={lang.code} value={lang.name}>
                        {lang.name}
                      </option>
                    ))}
                  </Select>
                </div>
                
                <div style={{ flex: 1 }}>
                  <Label style={{ marginBottom: '0.5rem' }}>翻译场景</Label>
                  <Select
                    value={selectedScene}
                    onChange={(e) => handleSceneChange(e.target.value)}
                    disabled={isTranslating}
                    style={{ width: '100%' }}
                    title={scenes.find(s => s.id === selectedScene)?.description || ''}
                  >
                    {scenes.map((scene) => (
                      <option key={scene.id} value={scene.id}>
                        {scene.name}
                      </option>
                    ))}
                  </Select>
                </div>
                
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                  {(isTranslating || isStopping) && (
                    <>
                      <Button
                        onClick={handleStopTranslation}
                        disabled={isStopping}
                        style={{ 
                          padding: '0.5rem 1rem', 
                          fontSize: '0.85rem',
                          backgroundColor: isStopping ? '#9e9e9e' : '#ff5722',
                          color: 'white',
                          border: 'none',
                          borderRadius: '6px',
                          display: 'flex',
                          alignItems: 'center',
                          gap: '0.5rem',
                          minWidth: 'auto',
                          cursor: isStopping ? 'not-allowed' : 'pointer',
                          opacity: isStopping ? 0.7 : 1
                        }}
                        title={isStopping ? "正在停止..." : "停止翻译"}
                      >
                        {isStopping ? '⏸️ 停止中...' : '⏹️ 停止'}
                      </Button>
                      {isTranslating && !isStopping && <LoadingSpinner style={{ margin: 0 }} />}
                    </>
                  )}
                </div>
              </FlexRow>
              
              {/* 额外要求输入 */}
              <div>
                <Label style={{ marginBottom: '0.5rem' }}>额外要求 (可选)</Label>
                <RequirementsInput
                  value={requirements}
                  onChange={(e) => handleRequirementsChange(e.target.value)}
                  placeholder="输入特殊的翻译要求，如语气、风格、专业术语处理等..."
                  disabled={isTranslating}
                />
              </div>
            </div>
            
            {error && <ErrorMessage style={{ marginBottom: '1rem' }}>{error}</ErrorMessage>}
            
            <Label>翻译结果</Label>
            <LargeResultArea className="fast-translation">
              {outputText || (isTranslating ? '正在翻译...' : '翻译结果将在这里显示')}
            </LargeResultArea>
          </ContentCard>
        </FlexColumn>
      </FlexRow>

      {/* 草稿区域 */}
      <div style={{ height: '30%' }}>
        <ContentCard style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
          <FlexRow style={{ alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.75rem' }}>
            <FlexRow style={{ alignItems: 'center', gap: '0.5rem' }}>
              <Label style={{ margin: 0 }}>📝 草稿笔记</Label>
              <span style={{ fontSize: '0.8rem', color: '#999' }}>
                记录翻译想法、备注或临时内容
              </span>
            </FlexRow>
            
            <FlexRow style={{ gap: '0.5rem' }}>
              <Button
                onClick={copyToDraft}
                disabled={!outputText}
                style={{ 
                  padding: '0.4rem 0.8rem', 
                  fontSize: '0.8rem',
                  backgroundColor: outputText ? '#e3f2fd' : '#f5f5f5',
                  color: outputText ? '#1976d2' : '#999'
                }}
                title="将翻译结果复制到草稿"
              >
                📋 复制结果
              </Button>
              <Button
                onClick={copyDraftToInput}
                disabled={!draftText}
                style={{ 
                  padding: '0.4rem 0.8rem', 
                  fontSize: '0.8rem',
                  backgroundColor: draftText ? '#fff3e0' : '#f5f5f5',
                  color: draftText ? '#f57c00' : '#999'
                }}
                title="将草稿内容复制到输入框"
              >
                📤 复制到输入
              </Button>
              <Button
                onClick={clearDraft}
                disabled={!draftText}
                style={{ 
                  padding: '0.4rem 0.8rem', 
                  fontSize: '0.8rem',
                  backgroundColor: draftText ? '#ffebee' : '#f5f5f5',
                  color: draftText ? '#d32f2f' : '#999'
                }}
                title="清空草稿内容"
              >
                🗑️ 清空
              </Button>
            </FlexRow>
          </FlexRow>
          
          <DraftTextArea
            value={draftText}
            onChange={(e) => setDraftText(e.target.value)}
            placeholder="💡 在这里记录你的翻译草稿、想法或备注...&#10;&#10;✨ 小贴士：&#10;• 使用右上角按钮快速复制翻译结果&#10;• 可以将草稿内容快速复制到输入框&#10;• 内容会自动保存，刷新页面不会丢失"
          />
        </ContentCard>
      </div>
    </div>
  );
};

export default FastTranslation;