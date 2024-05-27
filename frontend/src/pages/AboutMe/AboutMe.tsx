// pages/AboutMe/AboutMe.tsx
import React from 'react';

// contents
import About from 'pages/AboutMe/About';
import Contact from 'pages/AboutMe/Contact';

import style from 'styles/Main.module.css';

const AboutMe: React.FC = () => {
    return (
        <main className={style.content}>
            <h1>Introduction</h1>
            <About />
            <Contact />
        </main>
    );
};

export default AboutMe;
