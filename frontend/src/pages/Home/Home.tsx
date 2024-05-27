// HOme.tsx
import React from 'react';
import style from 'styles/Home.module.css';
import Problem from 'pages/Home/Problem';
import About from 'pages/Home/About';
import Contact from 'pages/Home/Contact';

import {Link} from 'react-router-dom';


const Main: React.FC = () => {
    return (
        <main className={style.content}>
            <h1>Project Introduction</h1>
            <Link to='service'></Link>
            <Problem />
            <About />
            <Contact />
        </main>
    );
};

export default Main;
