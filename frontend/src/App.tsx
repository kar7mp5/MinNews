import React, { useState } from 'react';
import 'App.css';

import Main from 'pages/Main';

const App: React.FC = () => {
    const [isSidebarOpen, setIsSidebarOpen] = useState<boolean>(false);

    const toggleSidebar = () => {
        setIsSidebarOpen(!isSidebarOpen);
    };

    return (
        <div className={`App ${isSidebarOpen ? 'sidebar-open' : ''}`}>
            <Main />
        </div>
    );
};

export default App;
