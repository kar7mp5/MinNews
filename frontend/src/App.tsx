import React, { useState } from 'react';
import 'App.css';
import Header from 'components/Header';
import Footer from 'components/Footer';
import Sidebar from 'components/Sidebar';
import Problem from 'components/Problem';
import About from 'components/About';
import Contact from 'components/Contact';

const App: React.FC = () => {
    const [isSidebarOpen, setIsSidebarOpen] = useState<boolean>(false);

    const toggleSidebar = () => {
        setIsSidebarOpen(!isSidebarOpen);
    };

    return (
        <div className={`App ${isSidebarOpen ? 'sidebar-open' : ''}`}>
            <Header isOpen={isSidebarOpen} toggleSidebar={toggleSidebar} />
            <Sidebar isOpen={isSidebarOpen} toggleSidebar={toggleSidebar} />
            <main className="content">
                <Problem />
                <About />
                <Contact />
            </main>
            <Footer />
        </div>
    );
};

export default App;
