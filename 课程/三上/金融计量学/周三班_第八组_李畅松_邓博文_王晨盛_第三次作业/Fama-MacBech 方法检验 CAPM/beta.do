clear
set mem 500m
cap log close 
log using capm.log, text replace
use "E:\jupyter_program\课程\金融计量学\TRD.dta", clear


*时间变量处理
replace Stkcd=substr(Stkcd,1,6)
gen date = date(Trddt,"YMD")
format date %td
rename date trddt
drop Trddt
sort trddt

***merge the stock data with index data by traddt****
merge trddt using "E:\jupyter_program\课程\金融计量学\SHA.dta"
tab _merge
keep if _merge==3
drop _merge

save "E:\jupyter_program\课程\金融计量学\temp_data.dta", replace

* 加载无风险收益率数据并排序
use "E:\jupyter_program\课程\金融计量学\rf.dta", clear
sort trddt
* 加载合并后的数据并再次排序
use "E:\jupyter_program\课程\金融计量学\temp_data.dta", clear
sort trddt


* 与无风险收益率合并
merge trddt using "E:\jupyter_program\课程\金融计量学\rf.dta"
 
tab _merge
keep if _merge == 3
drop _merge

***周数据*****
gen dow = dow(trddt)
gen year = year(trddt)
keep if year>2018&year<2023
gen friday = (dow==5)
sort Stkcd year trddt
by Stkcd year: gen week = sum(friday)
sort Stkcd year week trddt
by Stkcd year week: keep if friday==1

***生成周收益率*********
sort Stkcd trddt
*个股收益率ri = 本周股价 / 上周股价 - 1
*
by Stkcd: gen ri = Clsprc/Clsprc[_n-1]-1 if Stkcd==Stkcd[_n-1]
by Stkcd: gen rm = index/index[_n-1]-1 - rf if Stkcd==Stkcd[_n-1]

sort Stkcd year
by Stkcd year: keep if _N>=20

* 确保数据按时间顺序排序
sort trddt


****计算公司数量****
egen firm = group(Stkcd)
qui sum firm
local FIRM = r(max)

save "E:\jupyter_program\课程\金融计量学\capm.dta",replace

*估计每只股票的beta系数
use "E:\jupyter_program\课程\金融计量学\capm.dta",clear
*保存两个系数beta系数和标准误
statsby _b _se, by(firm): regress ri rm

histogram  _b_rm
histogram  _b_rm if  _b_rm<5
histogram  _b_rm if  _b_rm<5& _b_rm>0

save "E:\jupyter_program\课程\金融计量学\beta.dta",replace


* 将 firm 数据合并回原始数据集以计算平均收益率
use "E:\jupyter_program\课程\金融计量学\capm.dta", clear
* 计算每只股票的平均收益率
egen avg_ri = mean(ri), by(firm)

* 确保只有 firm 和 avg_ri 两个变量
keep firm avg_ri

* 在保存前进行排序，以保证唯一性
sort firm 

* 检查是否有重复的 firm
duplicates report firm

* 删除重复行，确保 firm 是唯一的
* 例如，选择性地保留第一个重复观察
by firm: keep if _n == 1

* 保存平均收益率数据
save "E:\jupyter_program\课程\金融计量学\avg_returns.dta", replace

use "E:\jupyter_program\课程\金融计量学\beta.dta", clear

* 合并平均收益率
merge 1:1 firm using "E:\jupyter_program\课程\金融计量学\avg_returns.dta"

* 检查合并是否成功
tab _merge

* 删除非匹配记录
keep if _merge == 3
drop _merge

gen _b_sq=_b_rm^2

twoway (sc avg_ri _b_rm) (qfit  avg_ri _b_rm)
reg avg_ri _b_rm _b_sq, r

* 打印回归结果，检查beta的显著性
display "Intercept: " _b[_cons]
display "Beta Coefficient: " _b[_b_rm]
