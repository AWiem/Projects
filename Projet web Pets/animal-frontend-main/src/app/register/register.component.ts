import { Component } from '@angular/core';
import { NonNullableFormBuilder, Validators } from '@angular/forms';
import { Router } from '@angular/router';
import { Client } from '../modals/Client';
import​ { HttpClient, HttpHeaders, HttpResponse } ​from​​'@angular/common/http'​;
import { AuthService } from '../auth.service';
@Component({
  selector: 'app-register',
  templateUrl: './register.component.html',
  styleUrls: ['./register.component.scss']
})
export class RegisterComponent {
  constructor(private router: Router,private fb: NonNullableFormBuilder,private authService:AuthService) {}
  registerError=false;

  //specific for client
  clientRegisterForm = this.fb.group({
    fullName:['', Validators.required],
    email: ['', [Validators.required, Validators.email]],
    address: ['', Validators.required],
    phone: ['', Validators.required],
    password: ['', Validators.required],
  });


  get clientFullName() {
    return this.clientRegisterForm.get('fullName');
  }

  get clientEmail() {
    return this.clientRegisterForm.get('email');
  }

  get clientAddress() {
    return this.clientRegisterForm.get('address');
  }

  get clientPhone() {
    return this.clientRegisterForm.get('phone');
  }

  get clientPassword() {
    return this.clientRegisterForm.get('password');
  }

  clientSubmit(){
    const client:Client = this.clientRegisterForm.value;
    this.registerError=false;
    console.log(client)
    this.authService.clientRegister(client).subscribe((response=> {
      let body=response.body;
      if(body){
        this.router.navigate(['/login']);
      }
    }),error=>{
      this.registerError=true;
    })
  }

  //specific for vet
  vetRegisterForm = this.fb.group({
    fullName:['', Validators.required],
    email: ['', [Validators.required, Validators.email]],
    password: ['', Validators.required],
  });


  //specific for pp
  ppRegisterForm = this.fb.group({
    fullName:['', Validators.required],
    email: ['', [Validators.required, Validators.email]],
    password: ['', Validators.required],
  });
}
