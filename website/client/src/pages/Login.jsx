import React, { useState } from "react";
import Button from "../components/Button";
import SvgComponent from "../components/SvgComponent";
import SignupForm from "../components/signup/SignUpForm";
import { useNavigate } from "react-router-dom";



const Login = () => {
  const [isSignupFormVisible, setIsSignupFormVisible] = useState(false);

  const navigate = useNavigate();

  const handleClick = async (purpose) => {
    if (purpose === "signup") {
      setIsSignupFormVisible(true);
    }
    if (purpose === "login") {
      navigate("/login");
    }
  };

  return (
    
    <>   
     <div className="navbar">
          <div className="typenavbar">
            <a href="#about-me" className="cursor-pointer">
              Home
            </a>
            <a href="#skills" className="cursor-pointer">
              Why AI Drawer?
            </a>
            <a href="#projects" className="cursor-pointer">
            About us
            </a>
          </div>
        </div>
        <header>
      <div className="logo-container">
      <div class="image-box"></div>
        <h1 className="company-name"> AI Drawer</h1>
  
  
      </div>
    </header>
    <div className="container">
      <div className="left-content">
        <h2 className="heading">Providing The Best </h2>
        <h2 className="heading2">AI Experience </h2>
        <p className="paragraph">
        Welcome to AIDrawer, where innovation meets intelligence! 
        Are you ready to experience the limitless possibilities of AI? 
        Whether you're seeking accurate answers, creative visuals, efficient code, or precise mathematical solutions,
         AIDrawer has got you covered. Our platform is designed to enhance your digital
         Let's dive in and explore the power of artificial intelligence together. 
         Are you curious about how AI can assist you today? Choose from our array 
         of tools and embark on a journey of exploration and innovation.
          Ask questions, generate images, receive coding assistance, or solve complex mathematical problems – the possibilities are endless!
        Join us now and unlock the full potential of AI with AIDrawer. 
        Your gateway to a smarter, more efficient digital world awaits!
        experience through a wide range of AI functionalities.

        </p>
      </div>
    </div>
      {!isSignupFormVisible ? (
        

       
            
           
            <div className="loginButtonWrapper">
              <Button text="Log in" handleClick={() => handleClick("login")} />
              <Button
                text="Sign up"
                handleClick={() => handleClick("signup")}
              />
            </div>
          
      ) : (
        <SignupForm />
      )}
      
    </>
  );
};

export default Login;
