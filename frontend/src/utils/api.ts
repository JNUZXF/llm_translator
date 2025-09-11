import { API_BASE_URL } from '../constants';
import { Language, TranslationScene, TranslationResponse, UploadResponse } from '../types';

export class APIError extends Error {
  constructor(message: string, public status?: number) {
    super(message);
    this.name = 'APIError';
  }
}

// 自定义EventSource-like类
class CustomEventSource {
  onmessage: ((event: MessageEvent) => void) | null = null;
  onerror: ((event: ErrorEvent) => void) | null = null;
  private reader: ReadableStreamDefaultReader<Uint8Array> | null = null;
  private closed = false;

  constructor(reader: ReadableStreamDefaultReader<Uint8Array>) {
    this.reader = reader;
    this.startReading();
  }

  private async startReading() {
    if (!this.reader) return;

    const decoder = new TextDecoder();
    
    try {
      while (!this.closed) {
        const { done, value } = await this.reader.read();
        if (done) break;
        
        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');
        
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.slice(6);
            if (data.trim() && this.onmessage) {
              const event = new MessageEvent('message', { data });
              this.onmessage(event);
            }
          }
        }
      }
    } catch (error) {
      if (this.onerror && !this.closed) {
        const errorEvent = new ErrorEvent('error', { error });
        this.onerror(errorEvent);
      }
    }
  }

  close() {
    this.closed = true;
    if (this.reader) {
      this.reader.cancel();
    }
  }

  addEventListener(type: string, listener: EventListenerOrEventListenerObject) {
    if (type === 'message') {
      this.onmessage = listener as (event: MessageEvent) => void;
    } else if (type === 'error') {
      this.onerror = listener as (event: ErrorEvent) => void;
    }
  }
}

export const api = {
  async cancelTranslation(sessionId: string): Promise<{ success: boolean; message: string }> {
    try {
      const response = await fetch(`${API_BASE_URL}/cancel-translation`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ session_id: sessionId }),
      });

      if (!response.ok) {
        throw new APIError(`HTTP ${response.status}: ${response.statusText}`, response.status);
      }

      return await response.json();
    } catch (error) {
      console.error('Cancel translation request failed:', error);
      if (error instanceof APIError) throw error;
      throw new APIError('取消翻译请求失败');
    }
  },
  async getLanguages(): Promise<Language[]> {
    try {
      const response = await fetch(`${API_BASE_URL}/languages`);
      if (!response.ok) {
        throw new APIError(`HTTP ${response.status}: ${response.statusText}`, response.status);
      }
      return await response.json();
    } catch (error) {
      if (error instanceof APIError) throw error;
      throw new APIError('获取语言列表失败');
    }
  },

  async getScenes(): Promise<TranslationScene[]> {
    try {
      const response = await fetch(`${API_BASE_URL}/scenes`);
      if (!response.ok) {
        throw new APIError(`HTTP ${response.status}: ${response.statusText}`, response.status);
      }
      return await response.json();
    } catch (error) {
      if (error instanceof APIError) throw error;
      throw new APIError('获取场景列表失败');
    }
  },

  async translateText(text: string, language: string, scene?: string, requirements?: string): Promise<EventSource> {
    try {
      const response = await fetch(`${API_BASE_URL}/translate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 
          text, 
          language, 
          scene: scene || 'ecommerce_amazon',
          requirements: requirements || ''
        }),
      });

      if (!response.ok) {
        throw new APIError(`HTTP ${response.status}: ${response.statusText}`, response.status);
      }

      // 检查响应类型
      const contentType = response.headers.get('content-type');
      if (!contentType || !contentType.includes('text/event-stream')) {
        throw new APIError('服务器返回了错误的内容类型');
      }

      if (!response.body) {
        throw new APIError('响应体为空');
      }

      const reader = response.body.getReader();
      return new CustomEventSource(reader) as unknown as EventSource;
    } catch (error) {
      console.error('Translation request failed:', error);
      throw error instanceof APIError ? error : new APIError('翻译请求失败');
    }
  },

  async uploadFile(file: File): Promise<UploadResponse> {
    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch(`${API_BASE_URL}/upload`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new APIError(`HTTP ${response.status}: ${response.statusText}`, response.status);
      }

      return await response.json();
    } catch (error) {
      console.error('File upload failed:', error);
      if (error instanceof APIError) throw error;
      throw new APIError('文件上传失败');
    }
  },

  async translatePaper(text: string): Promise<EventSource> {
    try {
      const response = await fetch(`${API_BASE_URL}/translate-paper`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text }),
      });

      if (!response.ok) {
        throw new APIError(`HTTP ${response.status}: ${response.statusText}`, response.status);
      }

      // 检查响应类型
      const contentType = response.headers.get('content-type');
      if (!contentType || !contentType.includes('text/event-stream')) {
        throw new APIError('服务器返回了错误的内容类型');
      }

      if (!response.body) {
        throw new APIError('响应体为空');
      }

      const reader = response.body.getReader();
      return new CustomEventSource(reader) as unknown as EventSource;
    } catch (error) {
      console.error('Paper translation request failed:', error);
      throw error instanceof APIError ? error : new APIError('论文翻译请求失败');
    }
  },

  async translatePDF(filepath: string): Promise<EventSource> {
    try {
      const response = await fetch(`${API_BASE_URL}/translate-pdf`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ filepath }),
      });

      if (!response.ok) {
        throw new APIError(`HTTP ${response.status}: ${response.statusText}`, response.status);
      }

      const contentType = response.headers.get('content-type');
      if (!contentType || !contentType.includes('text/event-stream')) {
        throw new APIError('服务器返回了错误的内容类型');
      }

      if (!response.body) {
        throw new APIError('响应体为空');
      }

      const reader = response.body.getReader();
      return new CustomEventSource(reader) as unknown as EventSource;
    } catch (error) {
      console.error('PDF translation request failed:', error);
      throw error instanceof APIError ? error : new APIError('PDF 翻译请求失败');
    }
  },

  async healthCheck(): Promise<{ status: string }> {
    try {
      const response = await fetch(`${API_BASE_URL}/health`);
      if (!response.ok) {
        throw new APIError(`HTTP ${response.status}: ${response.statusText}`, response.status);
      }
      return await response.json();
    } catch (error) {
      if (error instanceof APIError) throw error;
      throw new APIError('健康检查失败');
    }
  }
};