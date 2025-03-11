use "E:\jupyter_program\课程\金融计量学\任务二\399001_1.dta",clear
keep date rm
merge 1:1 date using "E:\jupyter_program\课程\金融计量学\任务二\珠江.dta"
keep if _merge==3
drop _merge
//2014-04-23 股东大会公告发布

estudy Return,datevar(date) evdate(20140423) dateformat(YMD) lb1(-5) ub1(-5) lb2(-4) ub2(-4) lb3(-3) ub3(-3) lb4(-2) ub4(-2) lb5(-1) ub5(-1) lb6(0) ub6(0) indexlist(rm) eswlb(-250) eswub(-77)

estudy Return,datevar(date) evdate(20140423) dateformat(YMD) lb1(-5) ub1(-5) lb2(-5) ub2(-4) lb3(-5) ub3(-3) lb4(-5) ub4(-2) lb5(-5) ub5(-1) lb6(-5) ub6(0) indexlist(rm) eswlb(-250) eswub(-77)

estudy Return,datevar(date) evdate(20140423) dateformat(YMD) lb1(1) ub1(1) lb2(2) ub2(2) lb3(3) ub3(3) lb4(4) ub4(4) lb5(5) ub5(5) indexlist(rm) eswlb(-250) eswub(-77)

estudy Return,datevar(date) evdate(20140423) dateformat(YMD) lb1(-5) ub1(1) lb2(-5) ub2(2) lb3(-5) ub3(3) lb4(-5) ub4(4) lb5(-5) ub5(5) indexlist(rm) eswlb(-250) eswub(-77)
