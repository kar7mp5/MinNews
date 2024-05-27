// src/components/Sidebar.tsx
import React from 'react';
import { Link } from 'react-router-dom';
import { FaTimes } from 'react-icons/fa';

import styles from 'styles/Sidebar.module.css';

interface SidebarProps {
    isOpen: boolean;
    toggleSidebar: () => void;
}

const Sidebar: React.FC<SidebarProps> = ({ isOpen, toggleSidebar }) => {
    return (
        <aside className={`${styles.sidebar} ${isOpen ? styles.open : ''}`}>
            <button className={styles.toggleButton} onClick={toggleSidebar}>
                <FaTimes className={styles.sidebarToggle} />
            </button>
            <nav>
                <ul className={styles.menu}>
                    <li>
                        <Link to="/">Home</Link>
                    </li>
                    <li>
                        <Link to="/service">Service</Link>
                    </li>
                    <li>
                        <Link to="/about">About Me</Link>
                    </li>
                </ul>
            </nav>
        </aside>
    );
};

export default Sidebar;
