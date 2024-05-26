// Body.tsx
import React from 'react';
import style from 'styles/Body.module.css';
import Problem from 'components/Problem';
import About from 'components/About';
import Contact from 'components/Contact';

const Body: React.FC = () => {
    return (
        <main className={style.content}>
            <h1>Project Introduction</h1>
            <Problem />
            <About />
            <Contact />
        </main>
    );
};

export default Body;
