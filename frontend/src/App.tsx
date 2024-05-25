import React, { useState } from 'react';
import 'App.css';
import Header from 'components/Header';
import Footer from 'components/Footer';
import Sidebar from 'components/Sidebar';
import Home from 'components/Home';
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
                <Home />
                <About />
                <Contact />
                Lorem ipsum dolor sit, amet consectetur adipisicing elit. Atque
                perferendis error aperiam excepturi nostrum, aliquid et omnis
                architecto maxime inventore, similique aut debitis, hic corporis
                sequi minima voluptatem qui facilis? Lorem ipsum dolor sit amet
                consectetur adipisicing elit. Non optio soluta modi, provident
                temporibus, dicta quam nemo, ad quod iusto officia animi at!
                Exercitationem, quod impedit? Cupiditate autem quasi sapiente?
                Lorem ipsum, dolor sit amet consectetur adipisicing elit.
                Adipisci voluptas hic, reprehenderit repellat iusto quibusdam
                fugit saepe? Minus corrupti officiis ea cumque dolorem illo,
                rem, harum id eius, perferendis neque? Lorem ipsum dolor sit
                amet consectetur adipisicing elit. Sit deleniti, nemo animi
                doloremque nesciunt reiciendis commodi architecto dolores rerum
                voluptatum in ratione explicabo nihil, nisi saepe dolorem ad
                corporis iusto. Lorem ipsum dolor, sit amet consectetur
                adipisicing elit. Natus fugiat eveniet eius dolorem autem ipsa
                facere assumenda ducimus vel, esse voluptate neque repudiandae
                inventore earum, magni ab dolores, nostrum suscipit? Lorem ipsum
                dolor sit amet consectetur adipisicing elit. Libero, sequi
                provident quos nostrum, aliquid cumque dolores dignissimos
                veniam, eaque hic inventore dolor ad voluptates ut illum
                consequatur quia minima laboriosam? Lorem ipsum dolor sit amet
                consectetur adipisicing elit. Architecto sapiente hic non
                suscipit fugit accusantium, rem esse aperiam numquam et autem
                at, excepturi aspernatur asperiores vel illum voluptate labore
                alias. Lorem ipsum, dolor sit amet consectetur adipisicing elit.
                Minus tempora molestiae, molestias architecto quisquam magni
                debitis accusamus quod consequatur, modi, voluptatibus mollitia
                autem enim. Debitis molestiae deserunt in dolorem fugit. Lorem
                ipsum dolor sit amet consectetur adipisicing elit. Corporis
                cupiditate dolor quidem obcaecati, velit enim! Tempora, quam
                delectus dolor, expedita ducimus unde quasi, nihil aliquid modi
                optio distinctio commodi iure? Lorem ipsum, dolor sit amet
                consectetur adipisicing elit. Fuga expedita odio, ratione earum
                nisi unde ipsum dolore ipsa hic impedit aliquam est quaerat
                blanditiis nemo nesciunt ea officiis distinctio. Reiciendis.
                Lorem ipsum dolor sit amet consectetur adipisicing elit. Nulla,
                modi accusamus aliquam consectetur officia voluptates laborum,
                ex vel velit libero tempora eligendi obcaecati ipsum aperiam.
                Earum aut tenetur aliquam maiores. Lorem ipsum dolor sit amet
                consectetur adipisicing elit. Consequatur nobis eaque ut? Ipsum
                amet temporibus ut! Nemo provident, recusandae suscipit
                repudiandae sapiente optio quos beatae veniam! Ad nam dolore a.
                Lorem ipsum dolor sit amet consectetur adipisicing elit.
                Doloribus, perferendis eius sint assumenda culpa, voluptatem non
                libero inventore cum dolorum ipsam mollitia provident sit
                reiciendis voluptates praesentium quod quibusdam tempora. Lorem
                ipsum dolor sit amet, consectetur adipisicing elit. Iusto nemo
                voluptatibus, sapiente magnam in excepturi reprehenderit
                deleniti, nobis eos fugit odio quidem animi expedita, voluptatem
                incidunt reiciendis aliquid voluptas dolores.
            </main>
            <Footer />
        </div>
    );
};

export default App;
