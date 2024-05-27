import React, { useState } from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import 'App.css';

import Sidebar from 'components/Sidebar';
import Header from 'components/Header';
import Footer from 'components/Footer';

// pages
import Home from 'pages/Home/Home';
import Service from 'pages/Service/Service';


const App: React.FC = () => {
    const [isSidebarOpen, setIsSidebarOpen] = useState<boolean>(false);

    const toggleSidebar = () => {
        setIsSidebarOpen(!isSidebarOpen);
    };

    return (
        <div className={`App ${isSidebarOpen ? 'sidebar-open' : ''}`}>
                        <BrowserRouter>

            <Sidebar isOpen={isSidebarOpen} toggleSidebar={toggleSidebar} />
            <Header isOpen={isSidebarOpen} toggleSidebar={toggleSidebar} />
				<Routes>
					<Route path="/" element={<Home />}></Route>
					<Route path="/service/*" element={<Service />}></Route>
					{/* 상단에 위치하는 라우트들의 규칙을 모두 확인, 일치하는 라우트가 없는경우 처리 */}
					<Route path="*" element={<Home />}></Route>
				</Routes>
            
            <Footer />
            </BrowserRouter>

        </div>
    );
};

export default App;