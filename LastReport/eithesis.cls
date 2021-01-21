%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%	長野高専電子情報工学科 卒業論文用クラスファイル
%
%   Original by Takuma Yoshida, 2010
%   Modified by Shoichi Ito, 2014-
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\typeout{Document Class: eithesis 2014/03/26 by Shoichi Ito (original by Takuma Yoshida).}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%	ベースとなる文書クラスの設定
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\ProcessOptions\relax
\LoadClass[a4paper,11pt,report]{jsbook}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%	必須パッケージの列挙
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\RequirePackage{listings, jlisting}
\RequirePackage{fancyvrb}
\RequirePackage[dvipdfmx]{graphicx}
\RequirePackage{booktabs}
\RequirePackage{amsmath,amssymb}	%数式

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%	ページのマージン設定(自分で設定する)
%	cf. http://www.wannabies.net/tex/howto/page.html
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%\setlength{\textwidth}{150mm}
%\setlength{\oddsidemargin}{4.6mm}
%\setlength{\evensidemargin}{4.6mm}
%\setlength{\topmargin}{4.6mm}
%\setlength{\headheight}{0mm}
%\setlength{\headsep}{0mm}
%\setlength{\footskip}{15mm}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%	表紙で使う変数の設定(ErotaThesisより接頭辞et)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\def\etTitle#1{\def\@etTitle{#1}}
\def\etTitleEn#1{\def\@etTitleEn{#1}}
\def\etLabName#1{\def\@etLabName{#1}}
\def\etMyProfessor#1{\def\@etMyProfessor{#1}}
\def\etDate#1{\def\@etDate{#1}}
\def\etAuthor#1{\def\@etAuthor{#1}}
\def\etGengou#1{\def\@etGengou{#1}}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%	文章中，論文タイトルと先生の名前を参照可能にする
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\def\etRefTitle{\@etTitle}
\def\etRefMyProfessor{\@etMyProfessor}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%	表紙の設定(vspace, vfillの前後には要改行)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\def\etMakeTitle{%
\begin{titlepage}
\centering

\vfill

\LARGE{自主課題研究 最終レポート}\\

\vspace*{10mm}

\Huge{\@etTitle}

\vspace{2mm}

\LARGE{\@etTitleEn}

\vfill

\Large{金沢大学 理工学域}\\
\Large{電子情報通信学類3年}\\
% \Large{\@etLabName} \\

\vspace{5mm}

\LARGE{\@etAuthor}

\vfill

% \Large{指導教員: \@etMyProfessor} \\

\vspace{2mm}

\Large{\@etDate}

\end{titlepage}
\pagestyle{plain} % ページ番号はいつもどおりにする
}%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%	各種コマンドの定義
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 脚注(footnote)を1)みたいな感じにする
\def\thefootnote{\ifnum\c@footnote>\z@\leavevmode\lower.5ex%
    \hbox{$^{\@arabic\c@footnote)}$}\fi}

% 章とタイトルの間に改行を入れない
\def\@makechapterhead#1{%
    \vspace*{2\Cvs} % 欧文は50pt
        {\parindent \z@ \raggedright \normalfont
        \Huge\headfont
        \ifnum \c@secnumdepth >\m@ne
            \if@mainmatter
                \@chapapp\thechapter\@chappos
                \hskip1zw
            \fi
        \fi
        #1\par\nobreak
        \vskip 3\Cvs}
}

%
\def\ps@lineheadings{\let\ps@jpl@in\ps@headnombre
    \let\@oddfoot\@empty\let\@evenfoot\@empty
    \def\@evenhead{%
        \underline{\hbox to\textwidth{\thepage\hfil\leftmark}}}%
    \def\@oddhead{%
        \underline{\hbox to\textwidth{\rightmark\hfil\thepage}}}%
    \let\@mkboth\markboth
    \def\chaptermark##1{\markboth{%
        \ifnum \c@secnumdepth >\m@ne%
            \@chapapp\thechapter\@chappos\hskip1zw
        \fi
    ##1}{}}%
    \def\sectionmark##1{\markright{%
        \ifnum \c@secnumdepth >\z@ \thesection.\hskip1zw\fi%
    ##1}}%
}

%  jlistings、C言語用の設定
%  cf. http://www.biwako.shiga-u.ac.jp/sensei/kumazawa/tex/listings.html
\lstset{
    language=C,                         %lstlisting環境内の言語の指定。参照。
    tabsize={4},                        %タブの展開後のサイズ
    numbers=left,                       %行番号表示。デフォルト：none。他のオプション：left、right。
    basicstyle={\small},                %書体の指定。行番号の書体指定。
    identifierstyle={\small},           %識別子の書体指定。
    %numberstyle=\scriptsize,           %行番号の書体指定。
    commentstyle={\small\ttfamily},     %注釈の書体。
    ndkeywordstyle={\small},            %
    keywordstyle={\small\bfseries},     %キーワードの書体指定。
    stringstyle={\small\ttfamily},      %
    columns=[l]{fullflexible},          %
    xrightmargin=0zw,                   %
    xleftmargin=0zw,                    %
    numbersep=1zw,                      %
    %backgroundcolor={\color[gray]{.85}},%
    frame=shadowbox,                    %frameの指定。デフォルト：none。他のオプション：leftline、topline、bottomline、lines、single、shadowbox。
    breaklines=true                     %行が長くなってしまった場合の改行。デフォルト：false。
}


%  実行結果表示用マクロ
%  第1引数は新しく作る環境名，第2引数は元となる環境名，第3引数は修正内容である．
%  http://www.biwako.shiga-u.ac.jp/sensei/kumazawa/tex/fancyvrb.html
\DefineVerbatimEnvironment{vresult}{Verbatim}{
    samepage=false,         %絶対同じページじゃなきゃ嫌なとき
    frame=lines,            %
    framesep=3mm,           %
    numbers=none,           %
    %fontfamily=helvetica,
    fontfamily=courier,     %これ大事、jsarticle.styを使用すると日本語フォントに対して太文字などが適用できない
    fontsize=\small,        %
    %xleftmargin=1cm        %
}

%  図とか表とかの参照用
\newcommand{\tabref}[1]{\tablename~\ref{#1}}
\newcommand{\equref}[1]{式~(\ref{#1})}
\newcommand{\figref}[1]{\figurename~\ref{#1}}

%  subsectionのがんばりを無駄にしない
\setcounter{tocdepth}{2}

%  章見出しが「第1章」のようになって「第」が邪魔なので消す
\renewcommand{\prechaptername}{}

%%% EOF %%%
