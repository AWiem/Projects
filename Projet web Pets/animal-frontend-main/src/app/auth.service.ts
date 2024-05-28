import { Injectable } from '@angular/core';
import​ { HttpClient, HttpHeaders, HttpResponse } ​from​​'@angular/common/http'​;
import { Client } from './modals/Client';
@Injectable({
  providedIn: 'root'
})
export class AuthService {

  baseUrl="http://localhost:7070/"
  constructor(private http: HttpClient) { }



  login(request:any){
    let body = `username=${request.email}&password=${request.password}`;

    const httpOptions = {
      headers: new HttpHeaders({ 'Content-Type': 'application/x-www-form-urlencoded' }),
      observe: 'response' as 'response',
      withCredentials:true
    };

    return this.http.post(this.baseUrl+"login",body,httpOptions);
  }

  clientRegister(client:Client){
    const httpOptions = {
      observe: 'response' as 'response',
      withCredentials:true
    };

    return this.http.post(this.baseUrl+"api/v1/registration",client,httpOptions)
  }

  logout(){
    return this.http.get(this.baseUrl+"api/v1/logout")
  }

  loggedIn():boolean{
    if(localStorage.getItem("email")){
      return true;
    }
    else{
      return false;
    }
  }

  getAccount(email:string){
    const httpOptions = {
      withCredentials:true
    };
    return this.http.get(this.baseUrl+"api/v1/clients/email/"+email,httpOptions);
  }
}
