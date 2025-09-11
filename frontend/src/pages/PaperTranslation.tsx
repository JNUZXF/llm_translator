import React, { useState, useRef, useEffect } from 'react';
import { 
  ContentCard, 
  ResultArea, 
  Button, 
  FlexRow, 
  FlexColumn, 
  Label,
  ErrorMessage,
  LoadingSpinner,
  FileUploadArea
} from '../styles/GlobalStyles';
import { api } from '../utils/api';
import { PDFPage, UploadResponse, TranslationResponse } from '../types';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeRaw from 'rehype-raw';
import rehypeHighlight from 'rehype-highlight';
import { API_BASE_URL } from '../constants';
import { usePersistedState } from '../hooks/usePersistedState';

const PaperTranslation: React.FC = () => {
  const [uploadedFile, setUploadedFile] = useState<UploadResponse | null>(null);
  const [selectedPage, setSelectedPage] = usePersistedState('paper-translation-selected-page', 0);
  const [translatedText, setTranslatedText] = useState('');
  // 保存文件基本信息用于恢复
  const [lastUploadedFileInfo, setLastUploadedFileInfo] = usePersistedState<{
    filename: string;
    pageCount: number;
    fileSize: number;
  } | null>('paper-translation-file-info', null);
  const [isUploading, setIsUploading] = useState(false);
  const [isTranslating, setIsTranslating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const leftPanelRef = useRef<HTMLDivElement>(null);

  const handleFileUpload = async (file: File) => {
    if (!file.type.includes('pdf')) {
      setError('只支持PDF文件');
      return;
    }

    setIsUploading(true);
    setError(null);

    try {
      const result = await api.uploadFile(file);
      setUploadedFile(result);
      setSelectedPage(0);
      setTranslatedText('');
      
      // 保存文件基本信息以便下次恢复
      setLastUploadedFileInfo({
        filename: result.filename,
        pageCount: result.info.page_count,
        fileSize: result.info.file_size
      });
      // 上传成功后，立即触发整份PDF翻译
      try {
        const eventSource = await api.translatePDF(result.filepath);

        eventSource.onmessage = (event) => {
          try {
            const data: TranslationResponse & { index?: number; total?: number; session_id?: string; cancelled?: boolean } = JSON.parse(event.data);

            if (data.session_id) {
              console.log('会话ID:', data.session_id);
              return;
            }

            if (data.error) {
              setError(data.error);
              setIsTranslating(false);
              eventSource.close();
              return;
            }

            if (data.cancelled) {
              setIsTranslating(false);
              eventSource.close();
              return;
            }

            if (data.content) {
              setTranslatedText(prev => prev + data.content + '\n\n');
            }

            if (data.done) {
              setIsTranslating(false);
              eventSource.close();
              // 翻译完毕后将左侧滚动定位到顶部
              setTimeout(() => {
                leftPanelRef.current?.scrollTo({ top: 0, behavior: 'smooth' });
              }, 0);
            }
          } catch (err) {
            console.error('解析响应数据失败:', err);
            setError('解析响应数据失败');
            setIsTranslating(false);
            eventSource.close();
          }
        };

        eventSource.onerror = (error) => {
          console.error('EventSource error:', error);
          setError('翻译服务连接失败');
          setIsTranslating(false);
          eventSource.close();
        };

        setIsTranslating(true);
      } catch (err) {
        console.error('自动触发PDF翻译失败:', err);
        setError(err instanceof Error ? err.message : '自动触发PDF翻译失败');
      }

    } catch (error) {
      setError(error instanceof Error ? error.message : '文件上传失败');
    } finally {
      setIsUploading(false);
    }
  };

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      handleFileUpload(file);
    }
  };

  const handleDrop = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setIsDragOver(false);
    
    const file = event.dataTransfer.files[0];
    if (file) {
      handleFileUpload(file);
    }
  };

  const handleDragOver = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = () => {
    setIsDragOver(false);
  };

  const handlePageTranslation = async (page: PDFPage) => {
    if (!page.text.trim()) {
      setError('该页面没有可翻译的文本');
      return;
    }

    setIsTranslating(true);
    setError(null);
    setTranslatedText('');

    try {
      const eventSource = await api.translatePaper(page.text);

      eventSource.onmessage = (event) => {
        try {
          const data: TranslationResponse = JSON.parse(event.data);
          
          if (data.error) {
            setError(data.error);
            setIsTranslating(false);
            eventSource.close();
            return;
          }

          if (data.content) {
            setTranslatedText(prev => prev + data.content);
          }

          if (data.done) {
            setIsTranslating(false);
            eventSource.close();
          }
        } catch (err) {
          console.error('解析响应数据失败:', err);
          setError('解析响应数据失败');
          setIsTranslating(false);
          eventSource.close();
        }
      };

      eventSource.onerror = (error) => {
        console.error('EventSource error:', error);
        setError('翻译服务连接失败');
        setIsTranslating(false);
        eventSource.close();
      };

    } catch (error) {
      console.error('翻译失败:', error);
      setError(error instanceof Error ? error.message : '翻译失败');
      setIsTranslating(false);
    }
  };

  const currentPage = uploadedFile?.pages[selectedPage];
  const pdfUrl = uploadedFile ? `${API_BASE_URL}/file/${encodeURIComponent(uploadedFile.filename)}` : '';

  return (
    <FlexRow $align="flex-start" style={{ height: '100%', gap: '2rem' }}>
      <FlexColumn>
        <ContentCard style={{ height: '100%' }}>
          <Label>PDF文件</Label>
          
          {!uploadedFile ? (
            <>
              {lastUploadedFileInfo && (
                <div style={{ 
                  marginBottom: '1rem', 
                  padding: '0.75rem', 
                  backgroundColor: 'rgba(102, 126, 234, 0.1)', 
                  borderRadius: '8px',
                  border: '1px solid rgba(102, 126, 234, 0.2)'
                }}>
                  <div style={{ fontSize: '0.9rem', color: '#666' }}>
                    <strong>上次使用的文件：</strong>{lastUploadedFileInfo.filename}
                  </div>
                  <div style={{ fontSize: '0.8rem', color: '#999', marginTop: '0.25rem' }}>
                    {lastUploadedFileInfo.pageCount} 页 • {(lastUploadedFileInfo.fileSize / 1024 / 1024).toFixed(1)}MB
                  </div>
                  <Button
                    onClick={() => setLastUploadedFileInfo(null)}
                    style={{ 
                      marginTop: '0.5rem', 
                      padding: '0.25rem 0.5rem', 
                      fontSize: '0.8rem',
                      backgroundColor: 'transparent',
                      color: '#666'
                    }}
                  >
                    清除历史记录
                  </Button>
                </div>
              )}
              <FileUploadArea
                $isDragOver={isDragOver}
                onDrop={handleDrop}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onClick={() => fileInputRef.current?.click()}
              >
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".pdf"
                  onChange={handleFileSelect}
                  style={{ display: 'none' }}
                />
                {isUploading ? (
                  <div>
                    <LoadingSpinner />
                    正在上传文件...
                  </div>
                ) : (
                  <div>
                    <p>点击选择PDF文件或拖拽文件到此处</p>
                    <p style={{ color: '#999', marginTop: '0.5rem' }}>
                      支持最大16MB的PDF文件
                    </p>
                  </div>
                )}
              </FileUploadArea>
            </>
          ) : (
            <div ref={leftPanelRef} style={{ display: 'flex', flexDirection: 'column', height: '100%', overflow: 'auto' }}>
              {/* PDF 原文预览（桌面工具效果，使用浏览器PDF查看器） */}
              <div style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
                <div style={{
                  border: '2px solid rgba(102, 126, 234, 0.2)',
                  borderRadius: '12px',
                  overflow: 'hidden',
                  background: 'rgba(255,255,255,0.8)',
                  height: '82vh'
                }}>
                  <iframe
                    title="pdf-preview"
                    src={pdfUrl}
                    style={{ width: '100%', height: '100%', border: 'none' }}
                  />
                </div>
              </div>

              {/* 基本文件信息（移至预览下方，保证顶部对齐） */}
              <div style={{ marginTop: '0.75rem' }}>
                <p><strong>文件名:</strong> {uploadedFile.filename}</p>
                <p><strong>页数:</strong> {uploadedFile.info.page_count}</p>
                <p><strong>文件大小:</strong> {(uploadedFile.info.file_size / 1024 / 1024).toFixed(2)} MB</p>
              </div>
            </div>
          )}
        </ContentCard>
      </FlexColumn>

      <FlexColumn>
        <ContentCard style={{ height: '100%' }}>
          <Label>翻译结果</Label>
          
          {error && <ErrorMessage>{error}</ErrorMessage>}
          
          <div style={{
            flex: 1,
            border: '2px solid rgba(102, 126, 234, 0.2)',
            borderRadius: '12px',
            background: 'rgba(255, 255, 255, 0.8)',
            padding: '1rem',
            overflowY: 'auto'
          }}>
            {translatedText ? (
              <ReactMarkdown
                remarkPlugins={[remarkGfm]}
                rehypePlugins={([rehypeRaw as any, (rehypeHighlight as unknown as any)] as any)}
                components={{
                  h1: ({ ...props }) => <h1 style={{ color: '#4a4a4a', borderBottom: '1px solid #eee', paddingBottom: '0.3rem', marginTop: '0.6rem' }} {...props} />,
                  h2: ({ ...props }) => <h2 style={{ color: '#555', marginTop: '0.6rem' }} {...props} />,
                  h3: ({ ...props }) => <h3 style={{ color: '#667eea', marginTop: '0.5rem' }} {...props} />,
                  ul: ({ ...props }) => <ul style={{ paddingLeft: '1.2rem', margin: '0.5rem 0' }} {...props} />,
                  ol: ({ ...props }) => <ol style={{ paddingLeft: '1.2rem', margin: '0.5rem 0' }} {...props} />,
                  code: ({ inline, ...props }: { inline?: boolean; [key: string]: any }) => (
                    <code style={{
                      background: 'rgba(102, 126, 234, 0.08)',
                      padding: inline ? '0.1rem 0.3rem' : '0.75rem',
                      borderRadius: '6px',
                      display: inline ? 'inline' : 'block',
                      overflowX: 'auto'
                    }} {...props} />
                  ),
                  blockquote: ({ ...props }) => (
                    <blockquote style={{
                      borderLeft: '3px solid #667eea',
                      paddingLeft: '0.75rem',
                      color: '#666',
                      margin: '0.5rem 0'
                    }} {...props} />
                  )
                }}
              >
                {translatedText}
              </ReactMarkdown>
            ) : (
              <div style={{ color: '#666' }}>
                {uploadedFile 
                  ? (isTranslating ? '正在翻译...' : '已上传PDF，等待翻译结果...') 
                  : '请先上传PDF文件'}
              </div>
            )}
          </div>
        </ContentCard>
      </FlexColumn>
    </FlexRow>
  );
};

export default PaperTranslation;