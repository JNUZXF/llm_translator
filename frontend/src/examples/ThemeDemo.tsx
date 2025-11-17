/**
 * 主题系统演示示例
 * 展示如何使用和扩展主题系统
 */

import React, { useState } from 'react';
import styled from 'styled-components';
import { useTheme, createTheme, lightTheme, Theme } from '../theme';
import { ThemeToggle, ThemeDropdown } from '../components/ThemeToggle';

// ============= 自定义主题示例 =============

// 示例1：基于亮色主题的海洋主题
const oceanTheme = createTheme(lightTheme, {
  name: 'Ocean',
  colors: {
    ...lightTheme.colors,
    semantic: {
      ...lightTheme.colors.semantic,
      primary: {
        main: '#0077be',
        light: '#0095e8',
        lighter: '#00b4ff',
        dark: '#005a92',
        darker: '#003f66',
        contrast: '#ffffff',
      },
    },
    background: {
      default: '#e6f3ff',
      paper: '#ffffff',
      elevated: '#f0f8ff',
      overlay: 'rgba(0, 119, 190, 0.5)',
    },
  },
});

// 示例2：森林主题
const forestTheme: Theme = {
  ...lightTheme,
  name: 'Forest',
  colors: {
    ...lightTheme.colors,
    semantic: {
      ...lightTheme.colors.semantic,
      primary: {
        main: '#2d5016',
        light: '#3d6821',
        lighter: '#4d7f2e',
        dark: '#1d3810',
        darker: '#0d2008',
        contrast: '#ffffff',
      },
    },
    background: {
      default: '#f0f5ed',
      paper: '#ffffff',
      elevated: '#f8faf6',
      overlay: 'rgba(45, 80, 22, 0.5)',
    },
  },
};

// ============= 演示组件 =============

export const ThemeDemo: React.FC = () => {
  const { theme } = useTheme();
  const [selectedPalette, setSelectedPalette] = useState<'primary' | 'secondary' | 'success'>('primary');

  return (
    <DemoContainer>
      <Header>
        <Title>主题系统演示</Title>
        <ThemeControls>
          <ThemeToggle showLabel />
          <ThemeDropdown />
        </ThemeControls>
      </Header>

      <Section>
        <SectionTitle>1. 颜色系统</SectionTitle>
        <ColorGrid>
          {/* 语义化颜色 */}
          {Object.entries(theme.colors.semantic).map(([name, palette]) => (
            <ColorCard key={name}>
              <ColorCardTitle>{name}</ColorCardTitle>
              <ColorSwatches>
                <ColorSwatch $color={palette.lighter} title="lighter" />
                <ColorSwatch $color={palette.light} title="light" />
                <ColorSwatch $color={palette.main} title="main" $large />
                <ColorSwatch $color={palette.dark} title="dark" />
                <ColorSwatch $color={palette.darker} title="darker" />
              </ColorSwatches>
            </ColorCard>
          ))}
        </ColorGrid>
      </Section>

      <Section>
        <SectionTitle>2. 排版系统</SectionTitle>
        <TypographyDemo>
          <Text $size="5xl" $weight="black">超大标题 (5xl / black)</Text>
          <Text $size="4xl" $weight="bold">大标题 (4xl / bold)</Text>
          <Text $size="3xl" $weight="bold">中标题 (3xl / bold)</Text>
          <Text $size="2xl" $weight="semibold">小标题 (2xl / semibold)</Text>
          <Text $size="xl" $weight="medium">正文大 (xl / medium)</Text>
          <Text $size="base" $weight="normal">正文 (base / normal)</Text>
          <Text $size="sm" $weight="normal">小字 (sm / normal)</Text>
          <Text $size="xs" $weight="normal">极小字 (xs / normal)</Text>
        </TypographyDemo>
      </Section>

      <Section>
        <SectionTitle>3. 间距系统</SectionTitle>
        <SpacingDemo>
          {[1, 2, 3, 4, 5, 6, 8, 10, 12, 16].map(size => (
            <SpacingBox key={size} $size={size}>
              {size}
            </SpacingBox>
          ))}
        </SpacingDemo>
      </Section>

      <Section>
        <SectionTitle>4. 阴影系统</SectionTitle>
        <ShadowGrid>
          {['xs', 'sm', 'md', 'lg', 'xl', '2xl'].map(shadow => (
            <ShadowCard key={shadow} $shadow={shadow as keyof Theme['shadows']}>
              {shadow}
            </ShadowCard>
          ))}
        </ShadowGrid>
      </Section>

      <Section>
        <SectionTitle>5. 圆角系统</SectionTitle>
        <RadiusGrid>
          {['xs', 'sm', 'md', 'lg', 'xl', '2xl', '3xl', 'full'].map(radius => (
            <RadiusBox key={radius} $radius={radius as keyof Theme['borderRadius']}>
              {radius}
            </RadiusBox>
          ))}
        </RadiusGrid>
      </Section>

      <Section>
        <SectionTitle>6. 按钮变体</SectionTitle>
        <ButtonGrid>
          <DemoButton $variant="primary">Primary Button</DemoButton>
          <DemoButton $variant="secondary">Secondary Button</DemoButton>
          <DemoButton $variant="success">Success Button</DemoButton>
          <DemoButton $variant="warning">Warning Button</DemoButton>
          <DemoButton $variant="error">Error Button</DemoButton>
          <DemoButton $variant="info">Info Button</DemoButton>
        </ButtonGrid>
      </Section>

      <Section>
        <SectionTitle>7. 卡片组件</SectionTitle>
        <CardsGrid>
          <DemoCard>
            <CardTitle>默认卡片</CardTitle>
            <CardContent>
              这是一个使用主题系统的卡片组件示例。
              它会自动适应当前主题。
            </CardContent>
          </DemoCard>
          <DemoCard $elevated>
            <CardTitle>悬浮卡片</CardTitle>
            <CardContent>
              这个卡片使用了elevated背景色，
              在暗色主题中会更加明显。
            </CardContent>
          </DemoCard>
          <DemoCard $bordered>
            <CardTitle>带边框卡片</CardTitle>
            <CardContent>
              这个卡片有明显的边框，
              展示了边框颜色的主题适配。
            </CardContent>
          </DemoCard>
        </CardsGrid>
      </Section>

      <Section>
        <SectionTitle>8. 表单元素</SectionTitle>
        <FormDemo>
          <Input type="text" placeholder="输入框" />
          <TextArea placeholder="文本域" rows={3} />
          <Select>
            <option>下拉选择</option>
            <option>选项 1</option>
            <option>选项 2</option>
          </Select>
        </FormDemo>
      </Section>

      <Section>
        <SectionTitle>9. 主题信息</SectionTitle>
        <InfoGrid>
          <InfoCard>
            <InfoLabel>主题模式</InfoLabel>
            <InfoValue>{theme.mode}</InfoValue>
          </InfoCard>
          <InfoCard>
            <InfoLabel>主题名称</InfoLabel>
            <InfoValue>{theme.name}</InfoValue>
          </InfoCard>
          <InfoCard>
            <InfoLabel>主色调</InfoLabel>
            <InfoValue>
              <ColorDot $color={theme.colors.semantic.primary.main} />
              {theme.colors.semantic.primary.main}
            </InfoValue>
          </InfoCard>
        </InfoGrid>
      </Section>
    </DemoContainer>
  );
};

// ============= 样式组件 =============

const DemoContainer = styled.div`
  max-width: 1200px;
  margin: 0 auto;
  padding: ${props => props.theme.spacing[8]};
  background: ${props => props.theme.colors.background.default};
  min-height: 100vh;
`;

const Header = styled.header`
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: ${props => props.theme.spacing[12]};
  padding-bottom: ${props => props.theme.spacing[8]};
  border-bottom: 2px solid ${props => props.theme.colors.border.default};
`;

const Title = styled.h1`
  font-size: ${props => props.theme.typography.fontSize['4xl']};
  font-weight: ${props => props.theme.typography.fontWeight.bold};
  color: ${props => props.theme.colors.text.primary};
  background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
  background-clip: text;
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
`;

const ThemeControls = styled.div`
  display: flex;
  gap: ${props => props.theme.spacing[4]};
  align-items: center;
`;

const Section = styled.section`
  margin-bottom: ${props => props.theme.spacing[12]};
`;

const SectionTitle = styled.h2`
  font-size: ${props => props.theme.typography.fontSize['2xl']};
  font-weight: ${props => props.theme.typography.fontWeight.semibold};
  color: ${props => props.theme.colors.text.primary};
  margin-bottom: ${props => props.theme.spacing[6]};
`;

const ColorGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: ${props => props.theme.spacing[4]};
`;

const ColorCard = styled.div`
  background: ${props => props.theme.colors.background.paper};
  padding: ${props => props.theme.spacing[4]};
  border-radius: ${props => props.theme.borderRadius.lg};
  border: 1px solid ${props => props.theme.colors.border.default};
`;

const ColorCardTitle = styled.h3`
  font-size: ${props => props.theme.typography.fontSize.sm};
  font-weight: ${props => props.theme.typography.fontWeight.semibold};
  color: ${props => props.theme.colors.text.secondary};
  margin-bottom: ${props => props.theme.spacing[2]};
  text-transform: capitalize;
`;

const ColorSwatches = styled.div`
  display: flex;
  gap: ${props => props.theme.spacing[1]};
  align-items: center;
`;

const ColorSwatch = styled.div<{ $color: string; $large?: boolean }>`
  width: ${props => props.$large ? '40px' : '30px'};
  height: ${props => props.$large ? '40px' : '30px'};
  background: ${props => props.$color};
  border-radius: ${props => props.theme.borderRadius.sm};
  border: 1px solid ${props => props.theme.colors.border.light};
  cursor: pointer;
  transition: transform ${props => props.theme.transitions.duration.fast};

  &:hover {
    transform: scale(1.1);
  }
`;

const TypographyDemo = styled.div`
  display: flex;
  flex-direction: column;
  gap: ${props => props.theme.spacing[2]};
  background: ${props => props.theme.colors.background.paper};
  padding: ${props => props.theme.spacing[6]};
  border-radius: ${props => props.theme.borderRadius.lg};
`;

const Text = styled.p<{ $size: keyof Theme['typography']['fontSize']; $weight: keyof Theme['typography']['fontWeight'] }>`
  font-size: ${props => props.theme.typography.fontSize[props.$size]};
  font-weight: ${props => props.theme.typography.fontWeight[props.$weight]};
  color: ${props => props.theme.colors.text.primary};
  margin: 0;
`;

const SpacingDemo = styled.div`
  display: flex;
  flex-wrap: wrap;
  gap: ${props => props.theme.spacing[2]};
  background: ${props => props.theme.colors.background.paper};
  padding: ${props => props.theme.spacing[6]};
  border-radius: ${props => props.theme.borderRadius.lg};
`;

const SpacingBox = styled.div<{ $size: number }>`
  width: ${props => props.theme.spacing[$size as keyof Theme['spacing']]};
  height: ${props => props.theme.spacing[$size as keyof Theme['spacing']]};
  background: ${props => props.theme.colors.semantic.primary.main};
  color: ${props => props.theme.colors.semantic.primary.contrast};
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: ${props => props.theme.borderRadius.sm};
  font-size: ${props => props.theme.typography.fontSize.xs};
  font-weight: ${props => props.theme.typography.fontWeight.bold};
`;

const ShadowGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: ${props => props.theme.spacing[6]};
`;

const ShadowCard = styled.div<{ $shadow: keyof Theme['shadows'] }>`
  background: ${props => props.theme.colors.background.paper};
  padding: ${props => props.theme.spacing[6]};
  border-radius: ${props => props.theme.borderRadius.lg};
  box-shadow: ${props => props.theme.shadows[props.$shadow]};
  text-align: center;
  font-weight: ${props => props.theme.typography.fontWeight.medium};
  color: ${props => props.theme.colors.text.primary};
`;

const RadiusGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(100px, 1fr));
  gap: ${props => props.theme.spacing[4]};
`;

const RadiusBox = styled.div<{ $radius: keyof Theme['borderRadius'] }>`
  background: ${props => props.theme.colors.semantic.primary.main};
  color: ${props => props.theme.colors.semantic.primary.contrast};
  padding: ${props => props.theme.spacing[4]};
  border-radius: ${props => props.theme.borderRadius[props.$radius]};
  text-align: center;
  font-size: ${props => props.theme.typography.fontSize.sm};
  font-weight: ${props => props.theme.typography.fontWeight.medium};
  min-height: 60px;
  display: flex;
  align-items: center;
  justify-content: center;
`;

const ButtonGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: ${props => props.theme.spacing[4]};
`;

const DemoButton = styled.button<{ $variant: keyof Theme['colors']['semantic'] }>`
  padding: ${props => props.theme.spacing[3]} ${props => props.theme.spacing[6]};
  border: none;
  border-radius: ${props => props.theme.borderRadius.md};
  background: ${props => props.theme.colors.semantic[props.$variant].main};
  color: ${props => props.theme.colors.semantic[props.$variant].contrast};
  font-size: ${props => props.theme.typography.fontSize.base};
  font-weight: ${props => props.theme.typography.fontWeight.medium};
  cursor: pointer;
  transition: all ${props => props.theme.transitions.duration.fast};

  &:hover {
    background: ${props => props.theme.colors.semantic[props.$variant].dark};
    transform: translateY(-2px);
    box-shadow: ${props => props.theme.shadows.lg};
  }

  &:active {
    transform: translateY(0);
  }
`;

const CardsGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: ${props => props.theme.spacing[6]};
`;

const DemoCard = styled.div<{ $elevated?: boolean; $bordered?: boolean }>`
  background: ${props => props.$elevated
    ? props.theme.colors.background.elevated
    : props.theme.colors.background.paper
  };
  padding: ${props => props.theme.spacing[6]};
  border-radius: ${props => props.theme.borderRadius.xl};
  border: ${props => props.$bordered
    ? `2px solid ${props.theme.colors.border.dark}`
    : `1px solid ${props.theme.colors.border.default}`
  };
  box-shadow: ${props => props.theme.shadows.md};
  transition: all ${props => props.theme.transitions.duration.normal};

  &:hover {
    box-shadow: ${props => props.theme.shadows.xl};
    transform: translateY(-4px);
  }
`;

const CardTitle = styled.h3`
  font-size: ${props => props.theme.typography.fontSize.lg};
  font-weight: ${props => props.theme.typography.fontWeight.semibold};
  color: ${props => props.theme.colors.text.primary};
  margin-bottom: ${props => props.theme.spacing[3]};
`;

const CardContent = styled.p`
  font-size: ${props => props.theme.typography.fontSize.sm};
  color: ${props => props.theme.colors.text.secondary};
  line-height: ${props => props.theme.typography.lineHeight.relaxed};
  margin: 0;
`;

const FormDemo = styled.div`
  display: flex;
  flex-direction: column;
  gap: ${props => props.theme.spacing[4]};
  max-width: 500px;
`;

const Input = styled.input`
  padding: ${props => props.theme.spacing[3]} ${props => props.theme.spacing[4]};
  border: 2px solid ${props => props.theme.colors.border.default};
  border-radius: ${props => props.theme.borderRadius.md};
  font-size: ${props => props.theme.typography.fontSize.base};
  background: ${props => props.theme.colors.background.paper};
  color: ${props => props.theme.colors.text.primary};
  transition: all ${props => props.theme.transitions.duration.fast};

  &:focus {
    outline: none;
    border-color: ${props => props.theme.colors.border.focus};
    box-shadow: 0 0 0 3px ${props => props.theme.colors.semantic.primary.lighter};
  }
`;

const TextArea = styled.textarea`
  padding: ${props => props.theme.spacing[3]} ${props => props.theme.spacing[4]};
  border: 2px solid ${props => props.theme.colors.border.default};
  border-radius: ${props => props.theme.borderRadius.md};
  font-size: ${props => props.theme.typography.fontSize.base};
  font-family: ${props => props.theme.typography.fontFamily.primary};
  background: ${props => props.theme.colors.background.paper};
  color: ${props => props.theme.colors.text.primary};
  resize: vertical;
  transition: all ${props => props.theme.transitions.duration.fast};

  &:focus {
    outline: none;
    border-color: ${props => props.theme.colors.border.focus};
    box-shadow: 0 0 0 3px ${props => props.theme.colors.semantic.primary.lighter};
  }
`;

const Select = styled.select`
  padding: ${props => props.theme.spacing[3]} ${props => props.theme.spacing[4]};
  border: 2px solid ${props => props.theme.colors.border.default};
  border-radius: ${props => props.theme.borderRadius.md};
  font-size: ${props => props.theme.typography.fontSize.base};
  background: ${props => props.theme.colors.background.paper};
  color: ${props => props.theme.colors.text.primary};
  cursor: pointer;
  transition: all ${props => props.theme.transitions.duration.fast};

  &:focus {
    outline: none;
    border-color: ${props => props.theme.colors.border.focus};
    box-shadow: 0 0 0 3px ${props => props.theme.colors.semantic.primary.lighter};
  }
`;

const InfoGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: ${props => props.theme.spacing[4]};
`;

const InfoCard = styled.div`
  background: ${props => props.theme.colors.background.paper};
  padding: ${props => props.theme.spacing[5]};
  border-radius: ${props => props.theme.borderRadius.lg};
  border: 1px solid ${props => props.theme.colors.border.default};
`;

const InfoLabel = styled.div`
  font-size: ${props => props.theme.typography.fontSize.sm};
  color: ${props => props.theme.colors.text.secondary};
  margin-bottom: ${props => props.theme.spacing[2]};
`;

const InfoValue = styled.div`
  font-size: ${props => props.theme.typography.fontSize.lg};
  font-weight: ${props => props.theme.typography.fontWeight.semibold};
  color: ${props => props.theme.colors.text.primary};
  display: flex;
  align-items: center;
  gap: ${props => props.theme.spacing[2]};
`;

const ColorDot = styled.div<{ $color: string }>`
  width: 16px;
  height: 16px;
  background: ${props => props.$color};
  border-radius: 50%;
  border: 1px solid ${props => props.theme.colors.border.default};
`;

export default ThemeDemo;
