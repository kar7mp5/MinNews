// pages/AboutMe/About.tsx
import React from 'react';
import MarkdownRenderer from 'components/MarkdownRenderer';
import ProfileImage from 'assets/Profile.png';

const About: React.FC = () => {
    return (
        <section id="about">
            <MarkdownRenderer
                markdownContent={`
## About Me
<img src="${ProfileImage}" alt="Profile Picture" width=20% height: auto/>

현재 데이터분석기초 수강 중인 인하대학교 컴퓨터공학과 12234073 김민섭입니다.
            `}
            />
        </section>
    );
};

export default About;
