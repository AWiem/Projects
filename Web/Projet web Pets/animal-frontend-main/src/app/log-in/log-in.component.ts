import { HttpResponse } from '@angular/common/http';
import { Component, OnInit } from '@angular/core';
import { NonNullableFormBuilder, Validators } from '@angular/forms';
import { Router } from '@angular/router';
import { AuthService } from '../auth.service';


@Component({
  selector: 'app-login',
  templateUrl: './log-in.component.html',
  styleUrls: ['./log-in.component.scss'],
})
export class LoginComponent implements OnInit {
  loginForm = this.fb.group({
    email: ['', [Validators.required, Validators.email]],
    password: ['', Validators.required],
  });

  constructor(private router: Router,private fb: NonNullableFormBuilder,private authService:AuthService) {}

  ngOnInit(): void {}

  loginError=false;

  get email() {
    return this.loginForm.get('email');
  }

  get password() {
    return this.loginForm.get('password');
  }

  submit() {
    const request = this.loginForm.value;
    this.loginError=false;
    if (!this.loginForm.valid || !request.email || !request.password) {
      return;
    }

    this.authService.login(request).subscribe((response: HttpResponse<any>)=>{
      let body=response.body;
      if(body['succsess']){
        console.log("Logged In");
        localStorage.setItem("email",request.email!);
        this.router.navigate(['/']);
      }else{
        this.loginError=true;
        console.log("Error")
      }
    },error=>{
      console.log(error)
    });

  }
}
