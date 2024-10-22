// src/components/Problem.tsx
import React from 'react';
import MarkdownRenderer from 'components/MarkdownRenderer';

const Problem: React.FC = () => {
    return (
        <section id="problem">
            <MarkdownRenderer
                markdownContent={`
## Problem
자취 후 TV를 통한 뉴스를 보기 힘들어지면서 인터넷으로 기사를
많이 접하게 되었습니다.  
TV에서도 과장하거나 잘못된 정보를 알려주는 경우가 종종 있었으나,
인터넷 기사에서는 내용과 상관없는 제목을 다는 경우를 심심치 않게
접하였습니다.  
인터넷 기사는 읽어야 하기 때문에 이와 같은 경우 시간이 낭비되었습니다.   
   
그렇다고 본문 내용이 질적으로 떨어지는 것도 아니었습니다.   
단지, 과장된 제목으로 인해 의도하지 않은 정보를 접하는 과정이 피곤했습니다.  
  
즉, “본문에 적절한 제목”을 제공하는 서비스 개발로 이 문제를
해결하고 싶었습니다.  
기존에 작성된 제목을 사용하지 않고, 본문 요약 및 수정을 통한 새로운 제목을 보여주는 방법을 고안하게
되었습니다.  
            `}
            />
        </section>
    );
};

export default Problem;
