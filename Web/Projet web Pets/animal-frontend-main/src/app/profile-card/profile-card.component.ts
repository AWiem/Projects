import { Component, OnInit } from '@angular/core';
import { MatDialog } from '@angular/material/dialog';
import { AuthService } from '../auth.service';
import { DialogBoxComponent } from '../dialog-box/dialog-box.component';
import { DialogFormComponent } from '../dialog-form/dialog-form.component';
import { Client } from '../modals/Client';

@Component({
  selector: 'app-profile-card',
  templateUrl: './profile-card.component.html',
  styleUrls: ['./profile-card.component.scss']
})
export class ProfileCardComponent implements OnInit{
  constructor(private authService:AuthService,public dialog:MatDialog){}
  client?:Client;
  ngOnInit(){
    let email=localStorage.getItem("email");
    this.authService.getAccount(email!).subscribe(response=>{
      this.client=response;
      console.log(this.client)
    })
  }

  openDialog(){
    this.dialog.open(DialogFormComponent,{
      width:'250px',
      data:"right click"
    })
  }

}
