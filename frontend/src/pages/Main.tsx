// pages/Main.tsx
import React, { useState } from 'react';

import Sidebar from 'components/Sidebar';
import Header from 'components/Header';
import Body from 'components/Body';
import Footer from 'components/Footer';

const App: React.FC = () => {
    const [isSidebarOpen, setIsSidebarOpen] = useState<boolean>(false);

    const toggleSidebar = () => {
        setIsSidebarOpen(!isSidebarOpen);
    };

    return (
        <div className={`App ${isSidebarOpen ? 'sidebar-open' : ''}`}>
            <Sidebar isOpen={isSidebarOpen} toggleSidebar={toggleSidebar} />
            <Header isOpen={isSidebarOpen} toggleSidebar={toggleSidebar} />
            <Body />
            <Footer />
        </div>
    );
};

export default App;
