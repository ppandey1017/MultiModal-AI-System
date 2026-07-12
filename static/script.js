const themeBtn = document.getElementById("themeBtn");

if(themeBtn){

    if(localStorage.getItem("theme")=="dark"){

        document.body.classList.add("dark");

        themeBtn.innerHTML='<i class="fas fa-sun"></i>';

    }

    themeBtn.onclick=function(){

        document.body.classList.toggle("dark");

        if(document.body.classList.contains("dark")){

            localStorage.setItem("theme","dark");

            themeBtn.innerHTML='<i class="fas fa-sun"></i>';

        }

        else{

            localStorage.setItem("theme","light");

            themeBtn.innerHTML='<i class="fas fa-moon"></i>';

        }

    };

}